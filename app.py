"""
LinAge2 biological age — Streamlit app.
Run locally:   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from linage2.score import LinAge2

ROOT = Path(__file__).parent
EXAMPLE = ROOT / "examples" / "example_input.csv"

INK, YOUNGER, OLDER, MUTED = "#1F3A4D", "#2E7D73", "#8A3B5C", "#9AA7B1"

st.set_page_config(page_title="LinAge2 biological age", page_icon="🧬", layout="wide")


@st.cache_resource
def load_clock():
    return LinAge2()


clock = load_clock()
LABELS = clock.p["labels"]
example = pd.read_csv(EXAMPLE).iloc[1]          # a 72-year-old man from NHANES 99/00
UNITS = {
    "SEQN": "—",
    "RIAGENDR": "1=Male, 2=Female",
    "RIDAGEEX": "months",
    "BPQ020": "1=Yes, 2=No",
    "DIQ010": "1=Yes, 2=No, 3=Borderline",
    "HUQ010": "1=Excellent ... 5=Poor",
    "HUQ020": "1=Better, 2=Worse, 3=About the same",
    "HUQ050": "1=None ... higher=more visits",
    "HUQ070": "1=Yes, 2=No",
    "KIQ020": "1=Yes, 2=No",
    "MCQ010": "1=Yes, 2=No",
    "MCQ053": "1=Yes, 2=No",
    "MCQ160A": "1=Yes, 2=No",
    "MCQ160B": "1=Yes, 2=No",
    "MCQ160C": "1=Yes, 2=No",
    "MCQ160D": "1=Yes, 2=No",
    "MCQ160E": "1=Yes, 2=No",
    "MCQ160F": "1=Yes, 2=No",
    "MCQ160G": "1=Yes, 2=No",
    "MCQ160I": "1=Yes, 2=No",
    "MCQ160J": "1=Yes, 2=No",
    "MCQ160K": "1=Yes, 2=No",
    "MCQ160L": "1=Yes, 2=No",
    "MCQ220": "1=Yes, 2=No",
    "OSQ010A": "1=Yes, 2=No",
    "OSQ010B": "1=Yes, 2=No",
    "OSQ010C": "1=Yes, 2=No",
    "OSQ060": "1=Yes, 2=No",
    "PFQ056": "1=Yes, 2=No",   # VERIFY
    "BPXPLS": "beats/min",
    "BPXSAR": "mmHg",
    "BPXDAR": "mmHg",
    "BMXBMI": "kg/m²",
    "URXUMASI": "mg/L",   # VERIFY
    "URXUCRSI": "µmol/L",   # VERIFY
    "LBDIRNSI": "µmol/L",
    "LBDTIBSI": "µmol/L",
    "LBXPCT": "%",
    "LBDFERSI": "µg/L",
    "LBDFOLSI": "nmol/L",
    "LBDB12SI": "pmol/L",
    "LBXCOT": "ng/mL",
    "LBDTCSI": "mmol/L",
    "LBDHDLSI": "mmol/L",
    "LBXWBCSI": "1000 cells/µL",
    "LBXLYPCT": "% of WBC",
    "LBXMOPCT": "% of WBC",
    "LBXNEPCT": "% of WBC",
    "LBXEOPCT": "% of WBC",
    "LBXBAPCT": "% of WBC",
    "LBDLYMNO": "1000 cells/µL",
    "LBDMONO": "1000 cells/µL",
    "LBDNENO": "1000 cells/µL",
    "LBDEONO": "1000 cells/µL",
    "LBDBANO": "1000 cells/µL",
    "LBXRBCSI": "million cells/µL",
    "LBXHGB": "g/dL",
    "LBXHCT": "%",
    "LBXMCVSI": "fL",
    "LBXMCHSI": "pg",
    "LBXMC": "g/dL",   # VERIFY
    "LBXRDW": "%",
    "LBXPLTSI": "1000 cells/µL",
    "LBXMPSI": "fL",
    "LBXCRP": "mg/dL",   # VERIFY
    "LBXGH": "%",
    "SSBNP": "pg/mL",   # VERIFY
    "LBDSALSI": "g/L",
    "LBXSATSI": "U/L",
    "LBXSASSI": "U/L",
    "LBXSAPSI": "U/L",
    "LBDSBUSI": "mmol/L",
    "LBDSCASI": "mmol/L",
    "LBXSC3SI": "mmol/L",
    "LBDSGLSI": "mmol/L",
    "LBXSLDSI": "U/L",
    "LBDSPHSI": "mmol/L",
    "LBDSTBSI": "µmol/L",
    "LBDSTPSI": "g/L",
    "LBDSTRSI": "mmol/L",
    "LBDSUASI": "µmol/L",
    "LBDSCRSI": "µmol/L",
    "LBXSNASI": "mmol/L",
    "LBXSKSI": "mmol/L",
    "LBXSCLSI": "mmol/L",
    "LBDSGBSI": "g/L",
}
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter


def build_template_xlsx(columns, example_row):
    """Three header rows: name, units, NHANES code. Row 3 is what score() reads."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    head_fill = PatternFill("solid", fgColor="1F3864")

    for j, v in enumerate(columns, start=1):
        c = ws.cell(1, j, LABELS.get(v, v))
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = head_fill
        c.alignment = Alignment(wrap_text=True, horizontal="center")

        ws.cell(2, j, UNITS.get(v, "")).font = Font(italic=True, size=9)
        ws.cell(3, j, v).font = Font(bold=True, size=9)
        ws.cell(4, j, example_row[v])          # one example row
        ws.column_dimensions[get_column_letter(j)].width = 15

    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "B4"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
# --------------------------------------------------------------------------
# Input layout
# --------------------------------------------------------------------------
GROUPS = {
    "Vitals": ["BPXPLS", "BPXSAR", "BPXDAR", "BMXBMI"],
    "Lipids": ["LBDTCSI", "LBDHDLSI", "LBDSTRSI"],
    "Blood count": ["LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
                    "LBDLYMNO", "LBDMONO", "LBDNENO", "LBDEONO", "LBDBANO", "LBXRBCSI",
                    "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI", "LBXMC", "LBXRDW",
                    "LBXPLTSI", "LBXMPSI"],
    "Chemistry": ["LBDSALSI", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBDSBUSI", "LBDSCASI",
                  "LBXSC3SI", "LBDSGLSI", "LBXSLDSI", "LBDSPHSI", "LBDSTBSI", "LBDSTPSI",
                  "LBDSUASI", "LBDSCRSI", "LBXSNASI", "LBXSKSI", "LBXSCLSI", "LBDSGBSI"],
    "Inflammation, glucose and heart": ["LBXCRP", "LBXGH", "SSBNP"],
    "Iron and vitamins": ["LBDIRNSI", "LBDTIBSI", "LBXPCT", "LBDFERSI", "LBDFOLSI", "LBDB12SI"],
    "Urine": ["URXUMASI", "URXUCRSI"],
}
YES_NO = ["BPQ020", "HUQ070", "KIQ020", "MCQ010", "MCQ053", "MCQ160A", "MCQ160B", "MCQ160C",
          "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K",
          "MCQ160L", "MCQ220", "OSQ010A", "OSQ010B", "OSQ010C", "OSQ060", "PFQ056"]
CHOICES = {
    "DIQ010": {1: "Yes", 2: "No", 3: "Borderline"},
    "HUQ010": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "HUQ020": {1: "Better", 2: "Worse", 3: "About the same"},
    "HUQ050": {0: "None", 1: "1", 2: "2–3", 3: "4–9", 4: "10–12", 5: "13 or more"},
}
SMOKING = {0: "Non-smoker", 1: "Light", 2: "Moderate", 3: "Heavy"}


def label(var):
    return LABELS.get(var, var)


def single_form():
    """Collect one person's values; returns (DataFrame row, cotinine mode)."""
    row = {"SEQN": 1}
    c1, c2, c3 = st.columns([1, 1, 2])
    sex = c1.radio("Sex", ["Male", "Female"], index=0 if example.RIAGENDR == 1 else 1, horizontal=True)
    row["RIAGENDR"] = 1 if sex == "Male" else 2
    age_y = c2.number_input("Age (years)", 18.0, 100.0, round(example.RIDAGEEX / 12, 1), 0.5)
    row["RIDAGEEX"] = age_y * 12
    if not 40 <= age_y <= 84:
        c3.warning("LinAge2 was trained on ages 40–84. Results outside that range are extrapolations.")

    with st.expander("Smoking", expanded=False):
        mode = st.radio("How do you want to enter smoking?",
                        ["Serum cotinine (ng/mL)", "Smoking intensity"], horizontal=True)
        if mode.startswith("Serum"):
            row["LBXCOT"] = st.number_input(label("LBXCOT"), 0.0, 2000.0, float(example.LBXCOT))
            cot = "C"
        else:
            row["LBXCOT"] = st.selectbox("Smoking intensity", list(SMOKING), format_func=SMOKING.get)
            cot = "S"

    for group, vars_ in GROUPS.items():
        with st.expander(group, expanded=False):
            cols = st.columns(3)
            for i, v in enumerate(vars_):
                row[v] = cols[i % 3].number_input(label(v), value=float(example[v]),
                                                  format="%.3f", key=v)

    with st.expander("Health history and self-rated health", expanded=False):
        cols = st.columns(2)
        for i, (v, opts) in enumerate(CHOICES.items()):
            keys = list(opts)
            default = int(example[v]) if int(example[v]) in keys else keys[0]
            row[v] = cols[i % 2].selectbox(label(v), keys, index=keys.index(default),
                                           format_func=opts.get, key=v)
        st.caption("Has a doctor ever told you that you have …  (tick for yes)")
        cols = st.columns(3)
        for i, v in enumerate(YES_NO):
            row[v] = 1 if cols[i % 3].checkbox(label(v), value=int(example[v]) == 1, key=v) else 2
    return pd.DataFrame([row]), cot


# --------------------------------------------------------------------------
# Result views
# --------------------------------------------------------------------------
def age_ruler(chron, bio):
    """The two ages on one horizontal scale."""
    lo, hi = min(chron, bio) - 8, max(chron, bio) + 8
    color = OLDER if bio > chron else YOUNGER
    pts = pd.DataFrame({"age": [chron, bio], "kind": ["Chronological", "LinAge2"],
                        "y": [0, 0]})
    span = pd.DataFrame({"a": [min(chron, bio)], "b": [max(chron, bio)], "y": [0]})
    base = alt.Chart(span).mark_rule(strokeWidth=6, color=color, opacity=0.35).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[lo, hi]), title="Age (years)"), x2="b", y=alt.value(40))
    dots = alt.Chart(pts).mark_circle(size=380, opacity=1).encode(
        x="age:Q", y=alt.value(40),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Chronological", "LinAge2"],
                                                  range=[MUTED, color]), legend=None),
        tooltip=["kind", alt.Tooltip("age:Q", format=".1f")])
    text = alt.Chart(pts).mark_text(dy=-24, fontSize=13, color=INK).encode(
        x="age:Q", y=alt.value(40), text=alt.Text("kind:N"))
    return (base + dots + text).properties(height=90)


def contribution_chart(contrib, n=12):
    s = contrib.drop("age_and_offset")
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(n)
    df = pd.DataFrame({"feature": [label(v) for v in top.index], "years": top.values})
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("years:Q", title="Years added (+) or removed (−) from LinAge2"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.condition("datum.years > 0", alt.value(OLDER), alt.value(YOUNGER)),
        tooltip=["feature", alt.Tooltip("years:Q", format="+.2f")],
    ).properties(height=28 * len(df))


def show_result(out, contrib):
    r = out.iloc[0]
    chron, bio, gap = r.chronAge, r.linAge2, r.bioAge_del
    word = "older" if gap > 0 else "younger"
    st.subheader(f"LinAge2: {bio:.1f} years")
    st.write(f"That is **{abs(gap):.1f} years {word}** than a chronological age of {chron:.1f}.")
    st.altair_chart(age_ruler(chron, bio), use_container_width=True)

    st.markdown("#### What moves the estimate")
    st.caption("LinAge2 is linear in the normalised inputs, so each value's share of the gap "
               "is exact. The remainder is the age term of the model.")
    st.altair_chart(contribution_chart(contrib), use_container_width=True)
    st.caption(f"Age term and model offset: {contrib['age_and_offset']:+.2f} years")

    with st.expander("Principal-component z-scores (vs. NHANES adults 40–50)"):
        pcs = r.filter(regex=r"^PC\d+[MF]$").dropna()
        st.bar_chart(pcs)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("LinAge2 biological age")
st.write("LinAge2 estimates biological age from routine clinical labs, vitals and a short "
         "health questionnaire. It was trained on NHANES 1999–2000 mortality follow-up and "
         "tested on NHANES 2001–2002.")

single, batch = st.tabs(["One person", "Many people (CSV)"])

with single:
    df, cot = single_form()
    if st.button("Calculate LinAge2", type="primary"):
        try:
            out = clock.score(df, cotinine=cot)
            contrib = clock.feature_contributions(df, cotinine=cot).iloc[0].drop("SEQN")
            show_result(out, contrib)
        except Exception as e:  # show the reason instead of a stack trace
            st.error(f"Could not calculate LinAge2: {e}")

with batch:
    st.write("Upload a CSV with NHANES column names, one person per row. "
             "Age (RIDAGEEX) is in months; sex (RIAGENDR) is 1 for male, 2 for female.")
    st.download_button("Download template", EXAMPLE.read_bytes(), "linage2_template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    cot_b = st.radio("The LBXCOT column holds", ["Cotinine (ng/mL)", "Smoking intensity 0–3"],
                     horizontal=True, key="cot_batch")
    if up is not None:
        try:
            data = pd.read_csv(up)
            res = clock.score(data, cotinine="C" if cot_b.startswith("Cot") else "S")
            show = res[["SEQN", "chronAge", "linAge2", "bioAge_del"]]
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button("Download results", res.to_csv(index=False).encode(),
                               "linage2_results.csv", "text/csv")
        except KeyError as e:
            st.error(f"The file is missing required columns. {e}")
        except Exception as e:
            st.error(f"Could not score the file: {e}")

st.divider()
st.caption("Research tool. Not a medical device and not medical advice. "
           "Inputs are processed in memory and not stored.")
