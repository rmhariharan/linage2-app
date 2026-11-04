# LinAge2 in Python

A Python port of the LinAge2 biological-age clock (original: `data/linAge2.R`),
with a Streamlit web app.

LinAge2 turns routine labs, vitals and a short health questionnaire into a
biological age. It is a sex-specific Cox proportional-hazards model on
principal components of normalised clinical features, trained on NHANES
1999–2000 mortality follow-up and tested on NHANES 2001–2002.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Score from Python:

```python
import pandas as pd
from linage2.score import LinAge2

clock = LinAge2()
out = clock.score(pd.read_csv("examples/example_input.csv"), cotinine="C")
print(out[["SEQN", "chronAge", "linAge2", "bioAge_del"]])
```

`cotinine="C"` if `LBXCOT` holds serum cotinine (ng/mL); `"S"` if it already
holds smoking intensity 0–3.

## How it is organised

| Path | What it does |
|---|---|
| `linage2/preprocess.py` | Column selection, frailty scores, LDL, albumin/creatinine ratio, filters, log transforms, z-scores, outlier folding |
| `linage2/cox.py` | Cox PH with Efron ties (numpy only), matching R `survival::coxph` |
| `linage2/train.py` | Full pipeline from NHANES → `linage2_params.json` |
| `linage2/linage2_params.json` | Frozen model: everything needed to score; the app needs nothing else |
| `linage2/score.py` | Scoring, PC z-scores, exact per-feature contributions |
| `app.py` | Streamlit app |
| `tests/test_linage2.py` | Verification suite |
| `data/` | NHANES training data, codebook, pre-computed SVD, original R script |

Rebuild the frozen model from raw data:

```bash
python -m linage2.train --data data --out linage2/linage2_params.json
```

## Verification

```bash
python tests/test_linage2.py      # or: python -m pytest -q
```

1. Cox fitter reproduces R's `coxph(Surv(time, status) ~ x, data = aml)` (coef 0.9155, se 0.5119).
2. The cleaned, normalised 1999/2000 data times V reproduces the pre-computed U·D to ~1e-13, for both sexes. This checks every preprocessing step end to end.
3. Cohort sizes (1045 M / 1034 F training) and scree cutoff (PC42) match the R script.
4. R's own sanity check: ten NHANES subjects re-entered as "user data" reproduce their training-set LinAge2.
5. Per-feature contributions add up exactly to LinAge2 − chronological age.

## Differences from the R script

- **SEQN 655 excluded from training.** With the supplied `mergedDataNHANES9902.csv`,
  1046 men pass the filters, but the pre-computed SVD has 1045 rows. Removing
  SEQN 655 makes the data reproduce the SVD exactly. (With these files the R
  script itself would stop at the step that writes U·D into the 1046-row matrix.)
- **Female PC labels.** R's `plotBars()` labels list PC32 twice and omit PC37.
  The Python output uses the model's actual PCs.
- Interactive prompts (`readline`) become function arguments; plots move to the app.
- The R ordering quirk where `MCQ160B` (heart failure) is read but not counted in the
  frailty index is kept, to stay faithful to the published clock.

## Deploying on Streamlit Community Cloud

Push the repo to GitHub, then create an app pointing at `app.py`. The app
loads only `linage2/linage2_params.json`; `data/` is needed only to retrain
(the NHANES CSV is ~7 MB, fine for GitHub).

Research tool. Not a medical device and not medical advice.
