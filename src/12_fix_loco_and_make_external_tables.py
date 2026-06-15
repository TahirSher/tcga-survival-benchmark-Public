from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project")
TABLES = ROOT / "outputs" / "tables"
MANUSCRIPT = TABLES / "manuscript"
EXTERNAL = ROOT / "outputs" / "external_validation" / "tables"

MANUSCRIPT.mkdir(parents=True, exist_ok=True)

def fmt4(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.4f}"
    except Exception:
        return ""

def fmt_p(x):
    try:
        if pd.isna(x):
            return ""
        x = float(x)
        if x < 1e-4:
            return f"{x:.2e}"
        return f"{x:.4f}"
    except Exception:
        return ""

def fmt_ci(lo, hi):
    try:
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f"{float(lo):.4f} to {float(hi):.4f}"
    except Exception:
        return ""

def fmt_hr(hr, lo, hi, status):
    status = str(status)
    try:
        if status != "penalized_cox" or pd.isna(hr) or pd.isna(lo) or pd.isna(hi):
            return "Not estimated"
        return f"{float(hr):.3f} ({float(lo):.3f} to {float(hi):.3f})"
    except Exception:
        return "Not estimated"

def save_latex_table(df, path, caption, label):
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format="l" * len(df.columns),
    )
    path.write_text(latex, encoding="utf-8")


# =============================================================================
# 1) Fix manuscript Table 3: Leave-one-cancer-out
# =============================================================================
loco_path = TABLES / "leave_one_cancer_out_results.csv"
if not loco_path.exists():
    raise FileNotFoundError(loco_path)

loco = pd.read_csv(loco_path)

required = {"holdout_cancer", "model_name", "holdout_test_cindex"}
missing = required - set(loco.columns)
if missing:
    raise ValueError(f"Missing required LOCO columns: {missing}")

model_order = [
    "Cox_ClinicalOnly",
    "Cox_FullAvailable",
    "ElasticNetCox_FullAvailable",
    "GradientBoostingSurvival_FullAvailable",
    "RandomSurvivalForest_FullAvailable",
    "SurvivalSVM_FullAvailable",
]

display = {
    "Cox_ClinicalOnly": "Cox (Clinical Only)",
    "Cox_FullAvailable": "Cox (Full Available)",
    "ElasticNetCox_FullAvailable": "Elastic Net Cox",
    "GradientBoostingSurvival_FullAvailable": "Gradient Boosting Survival",
    "RandomSurvivalForest_FullAvailable": "Random Survival Forest",
    "SurvivalSVM_FullAvailable": "Survival SVM",
}

rows = []
for model in model_order:
    g = loco.loc[loco["model_name"] == model].copy()
    row = {"Model": display[model]}

    vals = {}
    for cancer in ["BRCA", "COAD", "LUAD"]:
        sub = g.loc[g["holdout_cancer"] == cancer]
        val = float(sub["holdout_test_cindex"].iloc[0]) if len(sub) else np.nan
        vals[cancer] = val
        row[cancer] = fmt4(val)

    mean_val = np.nanmean([vals["BRCA"], vals["COAD"], vals["LUAD"]])
    row["Mean"] = fmt4(mean_val)
    rows.append(row)

table3 = pd.DataFrame(rows)
table3_csv = MANUSCRIPT / "table3_leave_one_cancer_out.csv"
table3_tex = MANUSCRIPT / "table3_leave_one_cancer_out.tex"
table3.to_csv(table3_csv, index=False)
save_latex_table(
    table3,
    table3_tex,
    caption="Leave-one-cancer-out external transfer within TCGA cohorts.",
    label="tab:loco",
)

print("\nFixed Table 3:")
print(table3.to_string(index=False))


# =============================================================================
# 2) External validation primary manuscript table
# =============================================================================
overall_path = EXTERNAL / "external_validation_overall_cindex.csv"
by_dataset_path = EXTERNAL / "external_validation_by_dataset_cindex.csv"
logrank_path = EXTERNAL / "external_risk_stratification_overall_logrank.csv"
hr_path = EXTERNAL / "external_risk_stratification_pairwise_hazard_ratios.csv"

for p in [overall_path, by_dataset_path, logrank_path, hr_path]:
    if not p.exists():
        raise FileNotFoundError(p)

overall = pd.read_csv(overall_path)
by_dataset = pd.read_csv(by_dataset_path)
logrank = pd.read_csv(logrank_path)
hr = pd.read_csv(hr_path)

primary_model = "Cox_ClinicalOnly"

primary = overall.loc[overall["model_name"] == primary_model].iloc[0]
primary_lr = logrank.loc[
    (logrank["model_name"] == primary_model)
    & (logrank["scope"] == "ALL_EXTERNAL")
].iloc[0]
primary_hr = hr.loc[
    (hr["model_name"] == primary_model)
    & (hr["scope"] == "ALL_EXTERNAL")
    & (hr["reference_group"] == "Low")
    & (hr["comparison_group"] == "High")
].iloc[0]

external_primary = pd.DataFrame([{
    "Model": "Cox (Clinical Only)",
    "External N": int(primary["n_patients"]),
    "Events": int(primary["n_events"]),
    "Harrell C-index": fmt4(primary["external_cindex"]),
    "Uno C-index": fmt4(primary["uno_cindex"]),
    "95% CI": fmt_ci(primary["ci_lower_95"], primary["ci_upper_95"]),
    "Log-rank p": fmt_p(primary_lr["overall_logrank_pvalue"]),
    "High vs Low HR": fmt_hr(
        primary_hr["hazard_ratio"],
        primary_hr["ci_lower_95"],
        primary_hr["ci_upper_95"],
        primary_hr["hr_status"],
    ),
    "HR p-value": fmt_p(primary_hr["p_value"]),
}])

external_primary_csv = MANUSCRIPT / "table_external_validation_primary.csv"
external_primary_tex = MANUSCRIPT / "table_external_validation_primary.tex"
external_primary.to_csv(external_primary_csv, index=False)
save_latex_table(
    external_primary,
    external_primary_tex,
    caption="External validation of the frozen TCGA-trained clinical Cox model.",
    label="tab:external_primary",
)

print("\nExternal primary table:")
print(external_primary.to_string(index=False))


# =============================================================================
# 3) External validation by cohort table
# =============================================================================
cohort_rows = []

cohort_order = [
    ("METABRIC", "BRCA"),
    ("GSE39582", "COAD"),
    ("GSE68465", "LUAD"),
]

for dataset, cancer in cohort_order:
    bd = by_dataset.loc[
        (by_dataset["model_name"] == primary_model)
        & (by_dataset["external_dataset"] == dataset)
        & (by_dataset["cancer_type"] == cancer)
    ].iloc[0]

    lr = logrank.loc[
        (logrank["model_name"] == primary_model)
        & (logrank["scope"] == dataset)
        & (logrank["cancer_type"] == cancer)
    ].iloc[0]

    hr_sub = hr.loc[
        (hr["model_name"] == primary_model)
        & (hr["scope"] == dataset)
        & (hr["cancer_type"] == cancer)
        & (hr["reference_group"] == "Low")
        & (hr["comparison_group"] == "High")
    ]

    if len(hr_sub):
        h = hr_sub.iloc[0]
        hr_text = fmt_hr(h["hazard_ratio"], h["ci_lower_95"], h["ci_upper_95"], h["hr_status"])
    else:
        hr_text = "Not estimated"

    cohort_rows.append({
        "External cohort": dataset,
        "Cancer": cancer,
        "N": int(bd["n_patients"]),
        "Events": int(bd["n_events"]),
        "Harrell C-index": fmt4(bd["external_cindex"]),
        "Uno C-index": fmt4(bd["uno_cindex"]),
        "Log-rank p": fmt_p(lr["overall_logrank_pvalue"]),
        "High vs Low HR": hr_text,
    })

external_by_cohort = pd.DataFrame(cohort_rows)
external_by_cohort_csv = MANUSCRIPT / "table_external_validation_by_cohort.csv"
external_by_cohort_tex = MANUSCRIPT / "table_external_validation_by_cohort.tex"
external_by_cohort.to_csv(external_by_cohort_csv, index=False)
save_latex_table(
    external_by_cohort,
    external_by_cohort_tex,
    caption="Cohort-wise external validation of the frozen TCGA-trained clinical Cox model.",
    label="tab:external_by_cohort",
)

print("\nExternal by-cohort table:")
print(external_by_cohort.to_string(index=False))


# =============================================================================
# 4) Simple validation
# =============================================================================
if table3[["BRCA", "COAD", "LUAD", "Mean"]].replace("", np.nan).isna().any().any():
    raise RuntimeError("Table 3 still contains missing values.")

print("\nSaved:")
print(" -", table3_csv)
print(" -", table3_tex)
print(" -", external_primary_csv)
print(" -", external_primary_tex)
print(" -", external_by_cohort_csv)
print(" -", external_by_cohort_tex)

print("\nDONE")
