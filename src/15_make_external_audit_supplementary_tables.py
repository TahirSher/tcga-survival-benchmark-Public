from pathlib import Path
import numpy as np
import pandas as pd

TCGA_ROOT = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project")
EXT_ROOT = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\External_Validation")

OUT = TCGA_ROOT / "outputs" / "tables" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

EXTERNAL_ALL = EXT_ROOT / "clean_external" / "external_master_stage1_like_all_rows.csv"
EXTERNAL_USABLE = EXT_ROOT / "clean_external" / "external_master_stage1_like_usable.csv"

OVERALL_CINDEX = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_validation_overall_cindex.csv"
BY_DATASET_CINDEX = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_validation_by_dataset_cindex.csv"
RISK_COUNTS = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_risk_stratification_group_counts.csv"
HR_TABLE = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_risk_stratification_pairwise_hazard_ratios.csv"

PRIMARY_MODEL = "Cox_ClinicalOnly"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def blank_mask(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().eq("")


def yes_no_count(nonmissing: int, total: int) -> str:
    if nonmissing <= 0:
        return "No"
    if nonmissing == total:
        return f"Yes ({nonmissing}/{total})"
    return f"Partial ({nonmissing}/{total})"


def fmt4(x) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.4f}"
    except Exception:
        return "--"


def fmt1(x) -> str:
    try:
        if pd.isna(x):
            return "--"
        return f"{float(x):.1f}"
    except Exception:
        return "--"


def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def latex_escape(s) -> str:
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def save_latex(df: pd.DataFrame, path: Path, caption: str, label: str, note: str, column_format: str) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize",
        r"\textit{Note:} " + latex_escape(note),
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Load
# =============================================================================

all_df = read_required(EXTERNAL_ALL)
usable_df = read_required(EXTERNAL_USABLE)
overall = read_required(OVERALL_CINDEX)
by_dataset = read_required(BY_DATASET_CINDEX)
risk_counts = read_required(RISK_COUNTS)
hr = read_required(HR_TABLE)

all_df["event_num"] = pd.to_numeric(all_df["event"], errors="coerce")
all_df["time_num"] = pd.to_numeric(all_df["time_to_event_days_raw"], errors="coerce")

usable_df["event"] = pd.to_numeric(usable_df["event"], errors="coerce")
usable_df["time_to_event_days_raw"] = pd.to_numeric(usable_df["time_to_event_days_raw"], errors="coerce")


# =============================================================================
# S8. External cohort exclusion flow
# =============================================================================

flow_rows = []

for (dataset, cancer), g in all_df.groupby(["external_dataset", "cancer_type"], dropna=False):
    raw_n = len(g)

    valid_event = g["event_num"].isin([0.0, 1.0])
    excluded_event = int((~valid_event).sum())

    g_event = g.loc[valid_event].copy()
    missing_time = g_event["time_num"].isna()
    excluded_missing_time = int(missing_time.sum())

    g_time = g_event.loc[~missing_time].copy()
    nonpositive_time = g_time["time_num"] <= 0
    excluded_nonpositive_time = int(nonpositive_time.sum())

    final_n = int((valid_event & g["time_num"].notna() & (g["time_num"] > 0)).sum())

    flow_rows.append({
        "External Cohort": dataset,
        "Cancer": cancer,
        "Raw N": raw_n,
        "Missing/Invalid Event": excluded_event,
        "Missing Survival Time": excluded_missing_time,
        "Non-positive Time": excluded_nonpositive_time,
        "Final Eligible N": final_n,
    })

table_s8 = pd.DataFrame(flow_rows).sort_values(["External Cohort", "Cancer"])
table_s8.to_csv(OUT / "supp_table_s8_external_cohort_exclusion_flow.csv", index=False)

save_latex(
    table_s8,
    OUT / "supp_table_s8_external_cohort_exclusion_flow.tex",
    caption="External cohort exclusion flow before frozen TCGA model validation.",
    label="tab:supp_external_exclusion_flow",
    note="Exclusions were applied sequentially after harmonizing each external cohort to the TCGA-stage1-like clinical schema. Patients were excluded if event status was missing or invalid, survival time was missing, or survival time was non-positive.",
    column_format="llrrrrr",
)


# =============================================================================
# S9. External clinical feature overlap
# =============================================================================

feature_rows = []

for (dataset, cancer), g in usable_df.groupby(["external_dataset", "cancer_type"], dropna=False):
    total = len(g)

    age_nonmissing = int(pd.to_numeric(g["age_years_raw"], errors="coerce").notna().sum())
    sex_nonmissing = int((~blank_mask(g["sex_raw"])).sum())
    race_nonmissing = int((~blank_mask(g["race_raw"])).sum())
    ethnicity_nonmissing = int((~blank_mask(g["ethnicity_raw"])).sum())
    site_nonmissing = int((~blank_mask(g["primary_site_raw"])).sum())
    stage_nonmissing = int((~blank_mask(g["stage_raw"])).sum())

    feature_rows.append({
        "External Cohort": dataset,
        "Cancer": cancer,
        "Eligible N": total,
        "Age": yes_no_count(age_nonmissing, total),
        "Sex": yes_no_count(sex_nonmissing, total),
        "Race": yes_no_count(race_nonmissing, total),
        "Ethnicity": yes_no_count(ethnicity_nonmissing, total),
        "Primary Site": yes_no_count(site_nonmissing, total),
        "Stage Group": yes_no_count(stage_nonmissing, total),
    })

table_s9 = pd.DataFrame(feature_rows).sort_values(["External Cohort", "Cancer"])
table_s9.to_csv(OUT / "supp_table_s9_external_feature_overlap.csv", index=False)

save_latex(
    table_s9,
    OUT / "supp_table_s9_external_feature_overlap.tex",
    caption="Clinical feature overlap between the TCGA clinical-only feature schema and the independent external cohorts.",
    label="tab:supp_external_feature_overlap",
    note="Availability is reported among externally eligible patients. Primary site was assigned from cohort identity during harmonization. Stage information was unavailable for METABRIC in the harmonized clinical table, which reduced direct feature overlap for BRCA external validation.",
    column_format="lllcccccc",
)


# =============================================================================
# S10. External Uno C-index setup
# =============================================================================

uno_rows = []

primary_overall = overall.loc[overall["model_name"] == PRIMARY_MODEL].iloc[0]
uno_rows.append({
    "Scope": "All external",
    "Cohort": "METABRIC + GSE39582 + GSE68465",
    "Cancer": "BRCA/COAD/LUAD",
    "N": safe_int(primary_overall["n_patients"]),
    "Events": safe_int(primary_overall["n_events"]),
    "Uno C-index": fmt4(primary_overall["uno_cindex"]),
    "Tau (days)": fmt1(primary_overall["uno_tau_days"]),
    "TCGA Reference N": safe_int(primary_overall["n_train_ref_rows"]),
    "External N after Tau": safe_int(primary_overall["n_test_rows_used"]),
})

primary_by = by_dataset.loc[by_dataset["model_name"] == PRIMARY_MODEL].copy()
for _, r in primary_by.iterrows():
    uno_rows.append({
        "Scope": "Cohort",
        "Cohort": r["external_dataset"],
        "Cancer": r["cancer_type"],
        "N": safe_int(r["n_patients"]),
        "Events": safe_int(r["n_events"]),
        "Uno C-index": fmt4(r["uno_cindex"]),
        "Tau (days)": fmt1(r["uno_tau_days"]),
        "TCGA Reference N": safe_int(r["n_train_ref_rows"]),
        "External N after Tau": safe_int(r["n_test_rows_used"]),
    })

table_s10 = pd.DataFrame(uno_rows)
table_s10.to_csv(OUT / "supp_table_s10_external_uno_setup.csv", index=False)

save_latex(
    table_s10,
    OUT / "supp_table_s10_external_uno_setup.tex",
    caption="Uno C-index setup for external validation of the frozen TCGA-trained clinical Cox model.",
    label="tab:supp_external_uno_setup",
    note="Uno's C-index used the frozen TCGA train-plus-validation survival outcomes as the inverse-probability-of-censoring weighting reference set. External evaluation was restricted to patients with observed time not exceeding tau.",
    column_format="lllrrrrrr",
)


# =============================================================================
# S11. External risk-group counts
# =============================================================================

risk_primary = risk_counts.loc[risk_counts["model_name"] == PRIMARY_MODEL].copy()

risk_primary["risk_group"] = pd.Categorical(
    risk_primary["risk_group"],
    categories=["Low", "Intermediate", "High"],
    ordered=True,
)

risk_primary = risk_primary.sort_values(["scope", "cancer_type", "risk_group"])

risk_rows = []
for _, r in risk_primary.iterrows():
    risk_rows.append({
        "Scope": r["scope"],
        "Cancer": r["cancer_type"],
        "Risk Group": r["risk_group"],
        "N": safe_int(r["n_patients"]),
        "Events": safe_int(r["n_dead"]),
        "Censored": safe_int(r["n_alive"]),
    })

table_s11 = pd.DataFrame(risk_rows)
table_s11.to_csv(OUT / "supp_table_s11_external_risk_group_counts.csv", index=False)

save_latex(
    table_s11,
    OUT / "supp_table_s11_external_risk_group_counts.tex",
    caption="External risk-group counts for the frozen TCGA-trained clinical Cox model.",
    label="tab:supp_external_risk_group_counts",
    note="Risk groups were assigned using tertile thresholds fixed from TCGA train-validation predictions. The imbalanced low-risk groups in GSE39582/COAD and GSE68465/LUAD explain why cohort-specific high-versus-low hazard ratios were not estimated for those cohorts.",
    column_format="lllrrr",
)


# =============================================================================
# HR skip reason table, small companion output
# =============================================================================

hr_primary = hr.loc[
    (hr["model_name"] == PRIMARY_MODEL)
    & (hr["reference_group"] == "Low")
    & (hr["comparison_group"] == "High")
].copy()

hr_out = hr_primary[[
    "scope",
    "cancer_type",
    "n_ref",
    "n_cmp",
    "n_events_ref",
    "n_events_cmp",
    "hazard_ratio",
    "ci_lower_95",
    "ci_upper_95",
    "p_value",
    "hr_status",
]].copy()

hr_out = hr_out.rename(columns={
    "scope": "Scope",
    "cancer_type": "Cancer",
    "n_ref": "Low N",
    "n_cmp": "High N",
    "n_events_ref": "Low Events",
    "n_events_cmp": "High Events",
    "hazard_ratio": "HR",
    "ci_lower_95": "CI Lower",
    "ci_upper_95": "CI Upper",
    "p_value": "p-value",
    "hr_status": "HR Status",
})

hr_out.to_csv(OUT / "supp_external_hr_skip_reasons.csv", index=False)


# =============================================================================
# Print final tables
# =============================================================================

print("\n" + "=" * 100)
print("SUPPLEMENTARY TABLE S8: EXTERNAL COHORT EXCLUSION FLOW")
print("=" * 100)
print(table_s8.to_string(index=False))

print("\n" + "=" * 100)
print("SUPPLEMENTARY TABLE S9: EXTERNAL FEATURE OVERLAP")
print("=" * 100)
print(table_s9.to_string(index=False))

print("\n" + "=" * 100)
print("SUPPLEMENTARY TABLE S10: EXTERNAL UNO SETUP")
print("=" * 100)
print(table_s10.to_string(index=False))

print("\n" + "=" * 100)
print("SUPPLEMENTARY TABLE S11: EXTERNAL RISK-GROUP COUNTS")
print("=" * 100)
print(table_s11.to_string(index=False))

print("\n" + "=" * 100)
print("HIGH-vs-LOW HR SKIP/ESTIMATION REASONS")
print("=" * 100)
print(hr_out.to_string(index=False))

print("\nSaved LaTeX tables:")
print(" -", OUT / "supp_table_s8_external_cohort_exclusion_flow.tex")
print(" -", OUT / "supp_table_s9_external_feature_overlap.tex")
print(" -", OUT / "supp_table_s10_external_uno_setup.tex")
print(" -", OUT / "supp_table_s11_external_risk_group_counts.tex")
print("\nDONE")
