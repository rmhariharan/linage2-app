"""
Score new subjects with LinAge2 using the frozen parameter file.

Input: a table with NHANES-named columns (same layout as userData.csv):
SEQN, RIAGENDR (1 = male, 2 = female), RIDAGEEX (age at exam, MONTHS),
the questionnaire items used for the frailty scores, and the lab/exam values.

    from linage2.score import LinAge2
    clock = LinAge2()                        # loads linage2_params.json
    out = clock.score(df, cotinine="C")      # "C" raw cotinine, "S" smoking 0-3
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocess as pp

DEFAULT_PARAMS = Path(__file__).with_name("linage2_params.json")
SEX = {1: "M", 2: "F"}


class LinAge2:
    def __init__(self, params_path=DEFAULT_PARAMS):
        self.p = json.loads(Path(params_path).read_text())
        self.features = self.p["features"]
        self.norm = pd.DataFrame(self.p["norm"]).T
        self.lam = pd.Series(self.p["transform"], dtype=float)

    # ------------------------------------------------------------------
    def required_columns(self):
        return list(dict.fromkeys(self.p["input_columns"] + self.p["qdata_columns"]))

    def prepare(self, df, cotinine="C"):
        """Raw table -> folded z-scores (n x 59), plus sex and age (months)."""
        df = df.copy()
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        if str(cotinine).upper() == "C":
            df = pp.digitize_cotinine(df)
        d = df[self.p["input_columns"]].astype(float).reset_index(drop=True)
        q = df.reset_index(drop=True)
        d = pp.add_derived(d, q)[["SEQN"] + self.features]
        t = pp.box_cox(d, self.lam)
        sex = q["RIAGENDR"].astype(int).to_numpy()
        if not np.isin(sex, [1, 2]).all():
            raise ValueError("RIAGENDR must be 1 (male) or 2 (female)")
        z = pp.fold_outliers(pp.zscore(t, sex, self.norm), self.p["z_max"])
        return d, z, sex, q["RIDAGEEX"].astype(float).to_numpy()

    def score(self, df, cotinine="C", pc_zscores=True):
        d, z, sex, age = self.prepare(df, cotinine)
        X = z[self.features].to_numpy()
        bio = np.full(len(X), np.nan)
        pc_out = {}
        for code, s in SEX.items():
            m = self.p["models"][s]
            rows = sex == code
            if not rows.any():
                continue
            pcs = X[rows] @ np.array(m["V"])                  # [projectToSVD]
            cols = [k - 1 for k in m["pcs"]]
            xf = np.column_stack([age[rows], pcs[:, cols]])
            lp_full = (xf - np.array(m["full"]["means"])) @ np.array(m["full"]["coef"])
            lp_null = (age[rows] - m["null"]["means"][0]) * m["null"]["coef"][0]
            bio[rows] = age[rows] + (lp_full - lp_null) / np.log(2) * m["mrdt_months"]  # [calcBioAge]
            if pc_zscores:
                zpc = (pcs - np.array(m["young_pc_mean"])) / np.array(m["young_pc_sd"])
                for k in m["pcs"]:
                    pc_out.setdefault(f"PC{k}{s}", np.full(len(X), np.nan))[rows] = zpc[:, k - 1]
        out = d.copy()
        out["SEQN"] = out["SEQN"].astype(int)
        out["chronAge"] = np.round(age / 12, 2)
        out["linAge2"] = np.round(bio / 12, 2)
        out["bioAge_del"] = np.round(out["linAge2"] - out["chronAge"], 2)
        for s in ("M", "F"):                                  # male PCs first, as in R
            for k in self.p["models"][s]["pcs"]:
                key = f"PC{k}{s}"
                if pc_zscores:
                    out[key] = pc_out.get(key, np.full(len(X), np.nan))
        return out

    def feature_contributions(self, df, cotinine="C"):
        """Exact, additive breakdown of the bio-age gap (years) into input features.

        LinAge2 is linear in the folded z-scores z, so
            gap = sum_j z_j * w_j  +  offset,   w = V[:, model PCs] @ beta_PCs * MRDT/ln2
        where offset collects the age terms and the Cox centring constants.
        Returns a DataFrame (one row per subject): feature columns + 'age_and_offset'.
        """
        d, z, sex, age = self.prepare(df, cotinine)
        Z = z[self.features].to_numpy()
        out = np.zeros((len(Z), len(self.features) + 1))
        for code, s in SEX.items():
            rows = sex == code
            if not rows.any():
                continue
            m = self.p["models"][s]
            scale = m["mrdt_months"] / np.log(2) / 12
            beta, means = np.array(m["full"]["coef"]), np.array(m["full"]["means"])
            V = np.array(m["V"])[:, [k - 1 for k in m["pcs"]]]
            w = V @ beta[1:]
            out[rows, :-1] = Z[rows] * w * scale
            off = (beta[0] * (age[rows] - means[0])
                   - m["null"]["coef"][0] * (age[rows] - m["null"]["means"][0])
                   - means[1:] @ beta[1:])
            out[rows, -1] = off * scale
        res = pd.DataFrame(out, columns=self.features + ["age_and_offset"])
        res.insert(0, "SEQN", d["SEQN"].astype(int))
        return res
