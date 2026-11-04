"""
LinAge2 training pipeline — Python port of linAge2.R (sections I-VI).

Builds the clock from NHANES 1999-2002 and exports every number needed for
scoring (feature transforms, normalisation, PC coordinates, Cox models) to a
single JSON file. The web app only needs that JSON, not the NHANES data.

    python -m linage2.train --data data --out linage2/linage2_params.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocess as pp
from .cox import CoxPH

# Final GLMNET-selected models from the R script (section V.iii)
MODEL_PCS = {
    "M": [1, 2, 5, 6, 8, 11, 15, 16, 17, 19, 24, 25, 27, 31, 33, 36, 42],
    "F": [1, 2, 4, 6, 11, 13, 20, 22, 23, 24, 28, 31, 32, 35, 37, 38, 39],
}
SEX_CODE = {"M": 1, "F": 2}

# NOTE: SEQN 655 (male, 1999/2000) passes every filter in this data file but is
# not part of the pre-computed SVD (uMatDat99_M_pre.csv has 1045 rows, the
# filtered data 1046). Removing it makes X.V == U.D to machine precision, so it
# is excluded from the training cohort to reproduce the published clock.
EXCLUDE_TRAIN_SEQN = [655]


def read_params(path):
    p = pd.read_csv(path)
    return dict(zip(p["pName"], p["pValue"]))


def load_inputs(data_dir):
    d = Path(data_dir)
    return dict(
        paras=read_params(d / "paraInit.csv"),
        master=pd.read_csv(d / "mergedDataNHANES9902.csv"),
        codebook=pd.read_csv(d / "codebook_linAge2.csv"),
        lam=pd.read_csv(d / "logNoLog.csv").iloc[1, 1:],    # R: read.csv(...)[2,]
        svd={s: dict(V=pd.read_csv(d / f"vMatDat99_{s}_pre.csv").to_numpy(),
                     U=pd.read_csv(d / f"uMatDat99_{s}_pre.csv").to_numpy(),
                     D=pd.read_csv(d / f"diagDat99_{s}_pre.csv").to_numpy())
             for s in ("M", "F")},
    )


def build(data_dir, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    inp = load_inputs(data_dir)
    P, cb, lam = inp["paras"], inp["codebook"], inp["lam"]
    report = {}

    # ---- I/II: select, derive, filter --------------------------------------
    master = pp.digitize_cotinine(inp["master"])
    d = pp.select_data(master, cb)
    q = pp.select_qdata(master, cb)
    if P["derivedFeatFlag"]:
        d = pp.add_derived(d, q)
    d, q = pp.filter_cohort(d, q, cb, P["lowerAgeLimit"], P["upperAgeLimit"], P["NAcutOffpercent"])
    keep = ~q["SEQN"].isin(EXCLUDE_TRAIN_SEQN).to_numpy()
    d, q = d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)
    features = d.columns[1:].tolist()
    log(f"[II] cohort: {len(d)} subjects x {len(features)} features")

    # ---- III: transform, normalise, fold -----------------------------------
    t = pp.box_cox(d, lam)
    norm = pp.norm_params(t, q)
    z = pp.fold_outliers(pp.zscore(t, q["RIAGENDR"].to_numpy(), norm), 6)
    X = z[features].to_numpy()
    yrs = q["yearsNHANES"].astype(int).to_numpy()
    sex = q["RIAGENDR"].to_numpy()
    train, test = yrs == 9900, yrs == 102

    # ---- IV: PC coordinates + scree cutoff ---------------------------------
    nsv = inp["svd"]["M"]["V"].shape[1]
    pcs = np.zeros((len(d), nsv))
    cut = {}
    for s, code in SEX_CODE.items():
        V, U, D = (inp["svd"][s][k] for k in "VUD")
        tr = train & (sex == code)
        UD = U @ D
        err = np.abs(X[tr] @ V - UD).max()          # check: data reproduces SVD
        report[f"svd_check_{s}"] = float(err)
        log(f"[IV] {s}: |X.V - U.D|max = {err:.2e}  ({tr.sum()} training subjects)")
        if err > 1e-8:
            raise RuntimeError("Training data do not reproduce the pre-computed SVD")
        pcs[tr] = UD                                 # R uses U.D for training rows
        pcs[test & (sex == code)] = X[test & (sex == code)] @ V
        sv = np.diag(D)
        scree = sv ** 2 / np.sum(sv ** 2)
        below = np.where(scree < P["PCcutOffpercent"] / 100)[0]
        cut[s] = int(min(below[0] + 1 if len(below) else nsv, nsv))
    max_pc = max(cut.values())
    log(f"[IV] scree cutoff  M={cut['M']}  F={cut['F']}  -> keep PC1..PC{max_pc}")
    pcs = pcs[:, :max_pc]

    # ---- V: Cox models ------------------------------------------------------
    age = q["RIDAGEEX"].to_numpy(float)              # months
    time = q["PERMTH_EXM"].to_numpy(float)
    event = q["MORTSTAT"].to_numpy(float)
    models, bio = {}, np.full(len(d), np.nan)
    for s, code in SEX_CODE.items():
        tr = train & (sex == code)
        if max(MODEL_PCS[s]) > max_pc:
            raise RuntimeError("Model uses a PC beyond the scree cutoff")
        cols = [k - 1 for k in MODEL_PCS[s]]
        names = ["chronAge"] + [f"PC{k}" for k in MODEL_PCS[s]]
        Xfull = np.column_stack([age, pcs[:, cols]])
        null = CoxPH().fit(age[tr, None], time[tr], event[tr], ["chronAge"])
        full = CoxPH().fit(Xfull[tr], time[tr], event[tr], names)
        # R refits keeping terms with p < pValCoxCut (= 1) -> same model
        assert all(r[5] < 1 for r in full.summary())
        mrdt = round(np.log(2) / null.coef_[0], 2)  # months (age is in months)

        # ---- VI: bio-age for every NHANES subject of this sex -------------
        sx = sex == code
        delta = (full.linear_predictor(Xfull[sx]) - null.linear_predictor(age[sx, None])) / np.log(2) * mrdt
        bio[sx] = age[sx] + delta

        # PC z-scores relative to young (<=50 y) 1999/2000 controls  [userDataOut]
        young = tr & (q["RIDAGEYR"].to_numpy() <= 50)
        models[s] = dict(
            pcs=MODEL_PCS[s],
            V=inp["svd"][s]["V"][:, :max_pc].tolist(),
            full=dict(names=names, coef=full.coef_.tolist(), means=full.means_.tolist(),
                      se=full.se_.tolist(), loglik=full.loglik_, n=full.n_, events=full.n_events_),
            null=dict(coef=null.coef_.tolist(), means=null.means_.tolist(), loglik=null.loglik_),
            mrdt_months=mrdt,
            young_pc_mean=pcs[young].mean(axis=0).tolist(),
            young_pc_sd=pcs[young].std(axis=0, ddof=1).tolist(),
        )
        log(f"[V]  {s}: n={full.n_} events={full.n_events_} MRDT={mrdt/12:.2f} y "
            f"beta_age(null)={null.coef_[0]:.6f}")

    params = dict(
        version="LinAge2-python-1.0",
        features=features,
        input_columns=pp.data_columns(cb),
        qdata_columns=["RIAGENDR", "RIDAGEEX"] + list(pp.FS1_ITEMS),
        transform={f: (None if pd.isna(lam[f]) else float(lam[f])) for f in features},
        norm=norm.loc[features].to_dict(orient="index"),
        skip_norm=pp.SKIP_NORM,
        z_max=6.0,
        max_pc=max_pc,
        models=models,
        labels={v: str(h).strip() for v, h in
                cb.drop_duplicates("Var", keep="last").set_index("Var")["Human"].items()
                if v in set(pp.data_columns(cb)) | set(["RIAGENDR", "RIDAGEEX"] + list(pp.FS1_ITEMS))},
    )
    results = pd.DataFrame(dict(SEQN=q["SEQN"].astype(int), wave=yrs, sex=sex,
                                chronAge=age / 12, linAge2=bio / 12))
    return params, results, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="linage2/linage2_params.json")
    ap.add_argument("--nhanes-out", default=None, help="optional CSV of NHANES bio-ages")
    a = ap.parse_args()
    params, results, _ = build(a.data)
    Path(a.out).write_text(json.dumps(params))
    print(f"Wrote {a.out}")
    if a.nhanes_out:
        results.to_csv(a.nhanes_out, index=False)
        print(f"Wrote {a.nhanes_out}")


if __name__ == "__main__":
    main()
