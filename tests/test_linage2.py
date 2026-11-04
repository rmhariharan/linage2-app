"""
Verification suite for the Python LinAge2 port.
Run:  python -m pytest -q      (or)      python tests/test_linage2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

from linage2.cox import CoxPH            # noqa: E402
from linage2.score import LinAge2        # noqa: E402
from linage2.train import build          # noqa: E402

_BUILD = None


def _build():
    global _BUILD
    if _BUILD is None:
        _BUILD = build(DATA, verbose=False)
    return _BUILD


def test_cox_matches_r_aml():
    """R: coxph(Surv(time, status) ~ x, data = aml) -> coef 0.9155, se 0.5119."""
    t = np.array([9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161, 5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45], float)
    e = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], float)
    x = np.array([0] * 11 + [1] * 12, float)[:, None]
    m = CoxPH().fit(x, t, e)
    assert round(m.coef_[0], 4) == 0.9155
    assert round(m.se_[0], 4) == 0.5119


def test_preprocessing_reproduces_svd():
    """Filtered, transformed, normalised 1999/2000 data times V must equal U.D."""
    _, _, report = _build()
    assert report["svd_check_M"] < 1e-10
    assert report["svd_check_F"] < 1e-10


def test_cohort_and_cutoff():
    params, results, _ = _build()
    assert len(params["features"]) == 59
    assert params["max_pc"] == 42
    train = results[results.wave == 9900]
    assert (train.sex == 1).sum() == 1045 and (train.sex == 2).sum() == 1034


def test_training_gap_centred():
    """Cox centring makes the mean bio-age gap ~0 in the training wave."""
    _, results, _ = _build()
    tr = results[results.wave == 9900]
    assert abs((tr.linAge2 - tr.chronAge).mean()) < 0.05


def test_sanity_samples():
    """R's built-in sanity check: re-scored NHANES subjects (SEQN + 100000)
    must reproduce their training-set LinAge2."""
    _, results, _ = _build()
    ref = results.set_index("SEQN")["linAge2"]
    out = LinAge2().score(pd.read_csv(DATA / "userData_sanity.csv"), cotinine="C")
    expected = ref.loc[out.SEQN - 100000].to_numpy()
    assert np.abs(out.linAge2.to_numpy() - expected).max() < 0.006   # 2-dp rounding
    assert round(np.corrcoef(out.linAge2, expected)[0, 1], 4) == 1


def test_frozen_params_match_fresh_build():
    """The shipped JSON equals a fresh build from the raw data."""
    import json
    params, _, _ = _build()
    shipped = json.loads((ROOT / "linage2" / "linage2_params.json").read_text())
    for s in ("M", "F"):
        np.testing.assert_allclose(shipped["models"][s]["full"]["coef"],
                                   params["models"][s]["full"]["coef"], rtol=1e-10)


def test_feature_contributions_sum_to_gap():
    """Per-feature contributions add up exactly to LinAge2 - chronological age."""
    _, results, _ = _build()
    ref = results.set_index("SEQN")
    df = pd.read_csv(DATA / "userData_sanity.csv")
    f = LinAge2().feature_contributions(df, cotinine="C")
    r = ref.loc[f.SEQN - 100000]
    gap = (r.linAge2 - r.chronAge).to_numpy()
    assert np.abs(f.iloc[:, 1:].sum(axis=1).to_numpy() - gap).max() < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
