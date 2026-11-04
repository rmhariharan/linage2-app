"""
Data selection, cleaning and derived features for LinAge2.

Each function mirrors a function in the original linAge2.R script
(names in brackets). Behaviour, including quirks, is kept identical so
that Python results reproduce the R results.
"""
import numpy as np
import pandas as pd

# Columns used to build derived features, then removed from the feature matrix
DROP_AFTER_DERIVED = ["LBDHDLSI", "LBDSTRSI", "LBDTCSI"]

# Columns that are NOT z-score normalised  [normAsZscores_99_young_mf: skipCols]
SKIP_NORM = ["fs1Score", "fs2Score", "fs3Score", "LBXCOT", "LBDBANO"]


# --------------------------------------------------------------------------
# Column selection
# --------------------------------------------------------------------------
def data_columns(codebook: pd.DataFrame) -> list:
    """Variables flagged Data == 1, in codebook order (SEQN first).
    [markIncsFromCodeBook + dropCols]"""
    cols = codebook.loc[codebook["Data"] == 1, "Var"].astype(str).tolist()
    if cols[0] != "SEQN":
        raise ValueError("First Data==1 variable in codebook must be SEQN")
    return cols


def select_data(df: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """Numeric/clinical data matrix.  [dropCols]"""
    cols = data_columns(codebook)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing data columns: {missing}")
    return df[cols].astype(float).reset_index(drop=True)


def select_qdata(df: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """Everything that is NOT a data column (questionnaire, demographics,
    mortality), with SEQN first.  [qDataMatGen]
    A column absent from the codebook defaults to flag 0 (included).
    If a variable appears several times in the codebook, the last flag wins."""
    flags = {}
    for v, f in zip(codebook["Var"].astype(str), codebook["Data"]):
        flags[v] = f
    keep = ["SEQN"] + [c for c in df.columns
                       if c not in ("SEQN", "Unnamed: 0") and flags.get(c, 0) == 0]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# Derived features
# --------------------------------------------------------------------------
def digitize_cotinine(df: pd.DataFrame) -> pd.DataFrame:
    """Bin cotinine (ng/mL) into smoking intensity 0-3.  [digiCot]
    <10 -> 0 (non-smoker), 10-100 -> 1, 100-200 -> 2, >=200 -> 3. NA stays NA."""
    df = df.copy()
    cot = df["LBXCOT"].astype(float).to_numpy()
    out = cot.copy()
    out[cot < 10] = 0
    out[(cot >= 10) & (cot < 100)] = 1
    out[(cot >= 100) & (cot < 200)] = 2
    out[cot >= 200] = 3
    df["LBXCOT"] = out
    return df


FS1_ITEMS = {  # variable: safe value used when NA
    "BPQ020": 2, "DIQ010": 2, "HUQ010": 3, "HUQ020": 3, "HUQ050": 0, "HUQ070": 2,
    "KIQ020": 2, "MCQ010": 2, "MCQ053": 2, "MCQ160A": 2, "MCQ160B": 2, "MCQ160C": 2,
    "MCQ160D": 2, "MCQ160E": 2, "MCQ160F": 2, "MCQ160G": 2, "MCQ160I": 2,
    "MCQ160J": 2, "MCQ160K": 2, "MCQ160L": 2, "MCQ220": 2, "OSQ010A": 2,
    "OSQ010B": 2, "OSQ010C": 2, "OSQ060": 2, "PFQ056": 2,
}


def fs1_score(q: pd.DataFrame) -> np.ndarray:
    """Frailty / comorbidity index: fraction of 22 yes/no items.  [popPCFIfs1]"""
    v = {k: q[k].astype(float).fillna(safe).to_numpy() for k, safe in FS1_ITEMS.items()}
    items = [
        v["BPQ020"] == 1, (v["DIQ010"] == 1) | (v["DIQ010"] == 3), v["KIQ020"] == 1,
        v["MCQ010"] == 1, v["MCQ053"] == 1, v["MCQ160A"] == 1, v["MCQ160C"] == 1,
        v["MCQ160D"] == 1, v["MCQ160E"] == 1, v["MCQ160F"] == 1, v["MCQ160G"] == 1,
        v["MCQ160I"] == 1, v["MCQ160J"] == 1, v["MCQ160K"] == 1, v["MCQ160L"] == 1,
        v["MCQ220"] == 1, v["OSQ010A"] == 1, v["OSQ010B"] == 1, v["OSQ010C"] == 1,
        v["OSQ060"] == 1, v["PFQ056"] == 1, v["HUQ070"] == 1,
    ]
    return np.column_stack(items).sum(axis=1) / 22.0


def fs2_score(q: pd.DataFrame) -> np.ndarray:
    """Self-rated health x trajectory.  [popPCFIfs2]
    HUQ010 fair(4)=2, poor(5)=4; HUQ020 better(1) x0.5, worse(2) x2."""
    h10 = q["HUQ010"].astype(float).fillna(3).to_numpy()
    h20 = q["HUQ020"].astype(float).fillna(3).to_numpy()
    a = (h10 == 4) * 2 + (h10 == 5) * 4
    d = 1 - (h20 == 1) * 0.5 + (h20 == 2)
    return a * d


def fs3_score(q: pd.DataFrame) -> np.ndarray:
    """Number of healthcare visits in past year (HUQ050).  [popPCFIfs3]"""
    h = q["HUQ050"].astype(float).fillna(0).to_numpy().copy()
    h[(h == 77) | (h == 99)] = 0
    return h


def ldl(d: pd.DataFrame) -> np.ndarray:
    """Friedewald LDL (mmol/L) = TC - TG/5 - HDL;  0 if any input is NA.  [populateLDL]"""
    tc, hdl, tg = (d[c].to_numpy(dtype=float) for c in ("LBDTCSI", "LBDHDLSI", "LBDSTRSI"))
    out = tc - tg / 5 - hdl
    out[np.isnan(tc) | np.isnan(hdl) | np.isnan(tg)] = 0.0
    return out


def alb_cr_ratio(d: pd.DataFrame) -> np.ndarray:
    """Urine albumin / creatinine ratio (mg/g).  [main script]"""
    return d["URXUMASI"].to_numpy(float) / (d["URXUCRSI"].to_numpy(float) * 1.1312e-4)


def add_derived(d: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Append fs1-3, LDLV and crAlbRat, then drop HDL, TG and TC.  [section II.ii]"""
    d = d.copy()
    d["fs1Score"] = fs1_score(q)
    d["fs2Score"] = fs2_score(q)
    d["fs3Score"] = fs3_score(q)
    d["LDLV"] = ldl(d)
    d["crAlbRat"] = alb_cr_ratio(d)
    return d.drop(columns=DROP_AFTER_DERIVED)


# --------------------------------------------------------------------------
# Row / column filters  (applied to NHANES reference data only)
# --------------------------------------------------------------------------
def drop_na_columns(d: pd.DataFrame, codebook: pd.DataFrame, pna_cut: float) -> pd.DataFrame:
    """Drop features with NA fraction >= pna_cut unless ForceInc == 1.  [dropNAcolumns]"""
    forced = set(codebook.loc[codebook["ForceInc"] == 1, "Var"].astype(str))
    frac = d.isna().mean()
    keep = [c for c in d.columns if frac[c] < pna_cut or c in forced]
    return d[keep]


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------
def box_cox(d: pd.DataFrame, lam: pd.Series) -> pd.DataFrame:
    """lambda NA -> unchanged; 0 -> natural log; else (x^l - 1)/l.  [boxCoxTransform]"""
    d = d.copy()
    for c in d.columns[1:]:
        if c not in lam.index:
            raise KeyError(f"No transform entry for {c} in logNoLog.csv")
        l = lam[c]
        if pd.isna(l):
            continue
        x = d[c].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            d[c] = np.log(x) if l == 0 else (x ** l - 1) / l
    return d


def r_mad(x: np.ndarray) -> float:
    """R's mad(): median absolute deviation scaled by 1.4826."""
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fold_outliers(z: pd.DataFrame, zmax: float = 6.0) -> pd.DataFrame:
    """Cap |z| at zmax.  [foldOutliers]"""
    z = z.copy()
    cols = z.columns[1:]
    z[cols] = z[cols].clip(lower=-zmax, upper=zmax)
    return z


def filter_cohort(d: pd.DataFrame, q: pd.DataFrame, codebook: pd.DataFrame,
                  age_lower: float, age_upper: float, pna_cut: float):
    """Apply the NHANES cohort filters in the same order as the R script
    (section II.ii-iii): missing exam age, accidental deaths, age bracket,
    NA-heavy features, then any subject with remaining NAs."""
    keep = q["RIDAGEEX"].notna().to_numpy()
    d, q = d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)

    cod = q["UCOD_LEADING"].fillna(0).to_numpy()            # [nonAccidDeathFlags]
    keep = cod != 4
    d, q = d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)

    age = q["RIDAGEYR"].to_numpy()                          # [selectAgeBracket]
    keep = (age >= age_lower) & (age <= age_upper)
    d, q = d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)

    d = drop_na_columns(d, codebook, pna_cut)               # [dropNAcolumns]

    keep = (d.isna().sum(axis=1) == 0).to_numpy()           # [getNonNARows]
    d, q = d[keep].reset_index(drop=True), q[keep].reset_index(drop=True)
    return d, q


def norm_params(d_ref: pd.DataFrame, q_ref: pd.DataFrame, young_max_age: float = 50):
    """Sex-specific median and MAD per feature, computed on the young (<= 50 y)
    1999/2000 NHANES reference subjects.  [normAsZscores_99_young_mf]
    Returns DataFrame indexed by feature with columns med_m, mad_m, med_f, mad_f."""
    sel = (q_ref["yearsNHANES"].astype(int) == 9900) & (q_ref["RIDAGEYR"] <= young_max_age)
    ref = d_ref[sel.to_numpy()]
    male = (q_ref.loc[sel, "RIAGENDR"] == 1).to_numpy()
    rows = {}
    for c in d_ref.columns[1:]:
        xm, xf = ref[c].to_numpy(float)[male], ref[c].to_numpy(float)[~male]
        rows[c] = dict(med_m=np.median(xm), mad_m=r_mad(xm),
                       med_f=np.median(xf), mad_f=r_mad(xf))
    return pd.DataFrame(rows).T


def zscore(d: pd.DataFrame, sex: np.ndarray, params: pd.DataFrame) -> pd.DataFrame:
    """(x - median_sex) / MAD_sex for every feature except SKIP_NORM."""
    z = d.copy()
    male = np.asarray(sex) == 1
    for c in d.columns[1:]:
        if c in SKIP_NORM:
            continue
        p = params.loc[c]
        x = d[c].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[c] = np.where(male, (x - p.med_m) / p.mad_m, (x - p.med_f) / p.mad_f)
    return z
