from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

TCGA_ROOT = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project")
EXT_ROOT = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\External_Validation")

OUT = TCGA_ROOT / "outputs" / "external_validation" / "audit"
OUT_TABLES = TCGA_ROOT / "outputs" / "tables" / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

external_all_path = EXT_ROOT / "clean_external" / "external_master_stage1_like_all_rows.csv"
external_usable_path = EXT_ROOT / "clean_external" / "external_master_stage1_like_usable.csv"

overall_cindex_path = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_validation_overall_cindex.csv"
by_dataset_cindex_path = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_validation_by_dataset_cindex.csv"
risk_counts_path = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_risk_stratification_group_counts.csv"
hr_path = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_risk_stratification_pairwise_hazard_ratios.csv"
logrank_path = TCGA_ROOT / "outputs" / "external_validation" / "tables" / "external_risk_stratification_overall_logrank.csv"
model_path = TCGA_ROOT / "outputs" / "models" / "Cox_ClinicalOnly.joblib"
bundle_path = TCGA_ROOT / "outputs" / "processed" / "preprocessed_data_bundle.joblib"
pooled_benchmark_path = TCGA_ROOT / "outputs" / "tables" / "pooled_benchmark_results.csv"

PRIMARY_MODEL = "Cox_ClinicalOnly"


def is_blank_series(s):
    return s.fillna("").astype(str).str.strip().eq("")


def valid_event_series(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.isin([0.0, 1.0])


def valid_time_series(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.notna() & (x > 0)


def fmt_bool_available(n):
    return "Yes" if int(n) > 0 else "No"


# =============================================================================
# 1) External cohort exclusion flow
# Sequential exclusion:
# raw N -> excluded missing/invalid event -> excluded missing survival time
# -> excluded non-positive survival time -> final eligible N
# =============================================================================

all_df = pd.read_csv(external_all_path, low_memory=False)
usable_df = pd.read_csv(external_usable_path, low_memory=False)

all_df["event_num"] = pd.to_numeric(all_df["event"], errors="coerce")
all_df["time_num"] = pd.to_numeric(all_df["time_to_event_days_raw"], errors="coerce")

flow_rows = []

for (dataset, cancer), g in all_df.groupby(["external_dataset", "cancer_type"], dropna=False):
    raw_n = len(g)

    event_valid = g["event_num"].isin([0.0, 1.0])
    excluded_missing_or_invalid_event = int((~event_valid).sum())

    g_after_event = g.loc[event_valid].copy()
    missing_time = g_after_event["time_num"].isna()
    excluded_missing_survival_time = int(missing_time.sum())

    g_after_time_present = g_after_event.loc[~missing_time].copy()
    non_positive_time = g_after_time_present["time_num"] <= 0
    excluded_non_positive_time = int(non_positive_time.sum())

    final_eligible = int((event_valid & g["time_num"].notna() & (g["time_num"] > 0)).sum())

    flow_rows.append({
        "external_dataset": dataset,
        "cancer_type": cancer,
        "raw_starting_n": int(raw_n),
        "excluded_missing_or_invalid_event": excluded_missing_or_invalid_event,
        "excluded_missing_survival_time": excluded_missing_survival_time,
        "excluded_non_positive_survival_time": excluded_non_positive_time,
        "final_eligible_n": final_eligible,
        "total_excluded": int(raw_n - final_eligible),
    })

flow = pd.DataFrame(flow_rows).sort_values(["external_dataset", "cancer_type"])
flow.to_csv(OUT / "external_cohort_exclusion_flow.csv", index=False)
flow.to_csv(OUT_TABLES / "audit_external_cohort_exclusion_flow.csv", index=False)


# =============================================================================
# 2) External Uno setup
# =============================================================================

overall = pd.read_csv(overall_cindex_path)
by_dataset = pd.read_csv(by_dataset_cindex_path)

uno_overall = overall.loc[overall["model_name"] == PRIMARY_MODEL].copy()
uno_by_dataset = by_dataset.loc[by_dataset["model_name"] == PRIMARY_MODEL].copy()

uno_overall_out = uno_overall[[
    "model_name",
    "scope",
    "n_patients",
    "n_events",
    "n_censored",
    "external_cindex",
    "uno_cindex",
    "uno_tau_days",
    "n_train_ref_rows",
    "n_test_rows_used",
]].copy() if "scope" in uno_overall.columns else uno_overall[[
    "model_name",
    "n_patients",
    "n_events",
    "n_censored",
    "external_cindex",
    "uno_cindex",
    "uno_tau_days",
    "n_train_ref_rows",
    "n_test_rows_used",
]].copy()

uno_by_dataset_out = uno_by_dataset[[
    "model_name",
    "external_dataset",
    "cancer_type",
    "n_patients",
    "n_events",
    "n_censored",
    "external_cindex",
    "uno_cindex",
    "uno_tau_days",
    "n_train_ref_rows",
    "n_test_rows_used",
]].copy()

uno_overall_out.to_csv(OUT / "external_uno_setup_overall.csv", index=False)
uno_by_dataset_out.to_csv(OUT / "external_uno_setup_by_dataset.csv", index=False)
uno_overall_out.to_csv(OUT_TABLES / "audit_external_uno_setup_overall.csv", index=False)
uno_by_dataset_out.to_csv(OUT_TABLES / "audit_external_uno_setup_by_dataset.csv", index=False)


# =============================================================================
# 3) Bootstrap protocol
# =============================================================================

bootstrap_protocol = pd.DataFrame([{
    "model_name": PRIMARY_MODEL,
    "metric": "Harrell C-index",
    "bootstrap_replicates_requested": 1000,
    "random_seed": 42,
    "confidence_level": "95%",
    "ci_scope_currently_computed": "overall_external_only",
    "cohort_wise_ci_currently_computed": "No",
    "n_bootstrap_valid_from_output": int(uno_overall["n_bootstrap_valid"].iloc[0]),
    "ci_lower_95": float(uno_overall["ci_lower_95"].iloc[0]),
    "ci_upper_95": float(uno_overall["ci_upper_95"].iloc[0]),
}])

bootstrap_protocol.to_csv(OUT / "external_bootstrap_protocol.csv", index=False)
bootstrap_protocol.to_csv(OUT_TABLES / "audit_external_bootstrap_protocol.csv", index=False)


# =============================================================================
# 4) External risk-group counts
# =============================================================================

risk_counts = pd.read_csv(risk_counts_path)
risk_counts_primary = risk_counts.loc[risk_counts["model_name"] == PRIMARY_MODEL].copy()

# Normalize column order if present
preferred_risk_cols = [
    "model_name",
    "scope",
    "cancer_type",
    "risk_group",
    "n_patients",
    "n_dead",
    "n_alive",
    "median_time_days",
    "mean_risk_score",
]
risk_cols = [c for c in preferred_risk_cols if c in risk_counts_primary.columns]
risk_counts_primary = risk_counts_primary[risk_cols].copy()

risk_counts_primary.to_csv(OUT / "external_risk_group_counts_primary_model.csv", index=False)
risk_counts_primary.to_csv(OUT_TABLES / "audit_external_risk_group_counts_primary_model.csv", index=False)

risk_counts_overall = risk_counts_primary.loc[risk_counts_primary["scope"] == "ALL_EXTERNAL"].copy()
risk_counts_by_cohort = risk_counts_primary.loc[risk_counts_primary["scope"] != "ALL_EXTERNAL"].copy()

risk_counts_overall.to_csv(OUT / "external_risk_group_counts_overall_primary_model.csv", index=False)
risk_counts_by_cohort.to_csv(OUT / "external_risk_group_counts_by_cohort_primary_model.csv", index=False)


# =============================================================================
# 5) HR skip reasons and exact counts
# =============================================================================

hr = pd.read_csv(hr_path)
hr_primary = hr.loc[
    (hr["model_name"] == PRIMARY_MODEL)
    & (hr["reference_group"] == "Low")
    & (hr["comparison_group"] == "High")
].copy()

hr_cols = [
    "model_name",
    "scope",
    "cancer_type",
    "reference_group",
    "comparison_group",
    "n_rows",
    "n_ref",
    "n_cmp",
    "n_events_ref",
    "n_events_cmp",
    "hazard_ratio",
    "ci_lower_95",
    "ci_upper_95",
    "p_value",
    "hr_status",
]
hr_primary = hr_primary[[c for c in hr_cols if c in hr_primary.columns]]
hr_primary.to_csv(OUT / "external_hr_high_vs_low_primary_model.csv", index=False)
hr_primary.to_csv(OUT_TABLES / "audit_external_hr_high_vs_low_primary_model.csv", index=False)


# =============================================================================
# 6) Feature overlap and clinical variable availability
# =============================================================================

feature_rows = []

for (dataset, cancer), g_raw in all_df.groupby(["external_dataset", "cancer_type"], dropna=False):
    g_eligible = usable_df.loc[
        (usable_df["external_dataset"] == dataset)
        & (usable_df["cancer_type"] == cancer)
    ].copy()

    row = {
        "external_dataset": dataset,
        "cancer_type": cancer,
        "eligible_n": int(len(g_eligible)),
        "age_available": fmt_bool_available(g_eligible["age_years_raw"].notna().sum()),
        "age_non_missing_n": int(g_eligible["age_years_raw"].notna().sum()),
        "sex_available": fmt_bool_available((~is_blank_series(g_eligible["sex_raw"])).sum()),
        "sex_non_missing_n": int((~is_blank_series(g_eligible["sex_raw"])).sum()),
        "race_available": fmt_bool_available((~is_blank_series(g_eligible["race_raw"])).sum()),
        "race_non_missing_n": int((~is_blank_series(g_eligible["race_raw"])).sum()),
        "ethnicity_available": fmt_bool_available((~is_blank_series(g_eligible["ethnicity_raw"])).sum()),
        "ethnicity_non_missing_n": int((~is_blank_series(g_eligible["ethnicity_raw"])).sum()),
        "primary_site_available": fmt_bool_available((~is_blank_series(g_eligible["primary_site_raw"])).sum()),
        "primary_site_non_missing_n": int((~is_blank_series(g_eligible["primary_site_raw"])).sum()),
        "stage_group_available": fmt_bool_available((~is_blank_series(g_eligible["stage_raw"])).sum()),
        "stage_non_missing_n": int((~is_blank_series(g_eligible["stage_raw"])).sum()),
        "stage_missing_n": int(is_blank_series(g_eligible["stage_raw"]).sum()),
    }

    feature_rows.append(row)

feature_overlap = pd.DataFrame(feature_rows).sort_values(["external_dataset", "cancer_type"])
feature_overlap.to_csv(OUT / "external_clinical_feature_overlap.csv", index=False)
feature_overlap.to_csv(OUT_TABLES / "audit_external_clinical_feature_overlap.csv", index=False)


# =============================================================================
# 7) Final artifact confirmation
# =============================================================================

artifact = joblib.load(model_path)
bundle = joblib.load(bundle_path)
pooled = pd.read_csv(pooled_benchmark_path)

pooled_primary = pooled.loc[pooled["model_name"] == PRIMARY_MODEL].iloc[0]

artifact_summary = {
    "primary_model_name": PRIMARY_MODEL,
    "model_artifact_path": str(model_path),
    "preprocessing_bundle_path": str(bundle_path),
    "model_family_from_artifact": artifact.get("model_family", ""),
    "feature_set_from_artifact": artifact.get("feature_set", ""),
    "n_features_from_artifact": len(artifact.get("feature_names", [])),
    "feature_names_from_artifact": artifact.get("feature_names", []),
    "risk_sign_applied_from_artifact": artifact.get("risk_sign_applied", None),
    "risk_sign_applied_from_pooled_benchmark": float(pooled_primary["risk_sign_applied"]),
    "external_score_orientation_changed": "No",
    "external_preprocessing_refit": "No",
    "external_model_refit": "No",
    "external_threshold_recalibration": "No",
    "reference_set_for_uno": "TCGA train + validation survival outcomes from preprocessed_data_bundle.joblib",
    "n_tcga_train_ref": int(len(bundle["splits"]["train"]["event"])),
    "n_tcga_val_ref": int(len(bundle["splits"]["val"]["event"])),
    "n_tcga_trainval_ref": int(len(bundle["splits"]["train"]["event"]) + len(bundle["splits"]["val"]["event"])),
}

with open(OUT / "external_primary_artifact_confirmation.json", "w", encoding="utf-8") as f:
    json.dump(artifact_summary, f, indent=2, ensure_ascii=False)

pd.DataFrame([{
    k: (json.dumps(v) if isinstance(v, list) else v)
    for k, v in artifact_summary.items()
}]).to_csv(OUT_TABLES / "audit_external_primary_artifact_confirmation.csv", index=False)


# =============================================================================
# Print everything clearly
# =============================================================================

print("\n" + "="*100)
print("1) EXTERNAL COHORT EXCLUSION FLOW")
print("="*100)
print(flow.to_string(index=False))

print("\n" + "="*100)
print("2) EXTERNAL UNO SETUP - OVERALL")
print("="*100)
print(uno_overall_out.to_string(index=False))

print("\n" + "="*100)
print("2b) EXTERNAL UNO SETUP - BY DATASET")
print("="*100)
print(uno_by_dataset_out.to_string(index=False))

print("\n" + "="*100)
print("3) BOOTSTRAP PROTOCOL")
print("="*100)
print(bootstrap_protocol.to_string(index=False))

print("\n" + "="*100)
print("4) RISK-GROUP COUNTS - OVERALL")
print("="*100)
print(risk_counts_overall.to_string(index=False))

print("\n" + "="*100)
print("4b) RISK-GROUP COUNTS - BY COHORT")
print("="*100)
print(risk_counts_by_cohort.to_string(index=False))

print("\n" + "="*100)
print("5) HIGH-vs-LOW HR REASONS AND COUNTS")
print("="*100)
print(hr_primary.to_string(index=False))

print("\n" + "="*100)
print("6) CLINICAL FEATURE OVERLAP")
print("="*100)
print(feature_overlap.to_string(index=False))

print("\n" + "="*100)
print("7) PRIMARY ARTIFACT CONFIRMATION")
print("="*100)
for k, v in artifact_summary.items():
    if k == "feature_names_from_artifact":
        print(k + ":", " | ".join(v))
    else:
        print(k + ":", v)

print("\nSaved audit outputs to:")
print(" -", OUT)
print(" -", OUT_TABLES)
print("\nDONE")
