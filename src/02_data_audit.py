from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project\configs\config.yaml")


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_json(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -----------------------------------------------------------------------------
# Validation and summaries
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "patient_id",
    "cancer_type",
    "event",
    "time_to_event_days_raw",
    "age_years_raw",
    "sex_raw",
    "stage_raw",
    "received_pharmaceutical_treatment_raw",
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master_table_stage1.csv: {missing}")


def add_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["event"] = safe_numeric(out["event"])
    out["time_to_event_days_raw"] = safe_numeric(out["time_to_event_days_raw"])
    out["age_years_raw"] = safe_numeric(out["age_years_raw"])
    out["received_pharmaceutical_treatment_raw"] = safe_numeric(
        out["received_pharmaceutical_treatment_raw"]
    )

    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["cancer_type"] = out["cancer_type"].astype(str).str.strip()

    if "time_to_event_source_raw" in out.columns:
        out["time_to_event_source_raw"] = out["time_to_event_source_raw"].fillna("").astype(str).str.strip()
    if "followup_time_source_raw" in out.columns:
        out["followup_time_source_raw"] = out["followup_time_source_raw"].fillna("").astype(str).str.strip()

    out["missing_patient_id"] = out["patient_id"].eq("") | out["patient_id"].eq("nan")
    out["duplicate_patient_id"] = out.duplicated(subset=["patient_id"], keep=False)
    out["missing_event"] = out["event"].isna()
    out["invalid_event"] = ~out["event"].isin([0.0, 1.0]) & out["event"].notna()
    out["missing_time"] = out["time_to_event_days_raw"].isna()
    out["non_positive_time"] = out["time_to_event_days_raw"].notna() & (
        out["time_to_event_days_raw"] <= 0
    )
    out["valid_time"] = out["time_to_event_days_raw"].notna() & (
        out["time_to_event_days_raw"] > 0
    )
    out["missing_age"] = out["age_years_raw"].isna()
    out["invalid_age"] = out["age_years_raw"].notna() & (
        (out["age_years_raw"] < 0) | (out["age_years_raw"] > 120)
    )
    out["missing_sex"] = out["sex_raw"].fillna("").astype(str).str.strip().eq("")
    out["missing_stage"] = out["stage_raw"].fillna("").astype(str).str.strip().eq("")
    out["missing_treatment"] = out["received_pharmaceutical_treatment_raw"].isna()

    out["usable_for_survival_modeling"] = (
        ~out["missing_patient_id"]
        & ~out["missing_event"]
        & ~out["invalid_event"]
        & ~out["missing_time"]
        & ~out["non_positive_time"]
    )

    out["time_to_event_years_raw"] = out["time_to_event_days_raw"] / 365.25

    return out


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "n_rows": int(len(df)),
        "n_unique_patients": int(df["patient_id"].nunique(dropna=True)),
        "n_duplicate_patient_rows": int(df["duplicate_patient_id"].sum()),
        "n_missing_patient_id": int(df["missing_patient_id"].sum()),
        "n_missing_event": int(df["missing_event"].sum()),
        "n_invalid_event": int(df["invalid_event"].sum()),
        "n_missing_time": int(df["missing_time"].sum()),
        "n_non_positive_time": int(df["non_positive_time"].sum()),
        "n_usable_for_survival_modeling": int(df["usable_for_survival_modeling"].sum()),
        "n_dead": int((df["event"] == 1.0).sum()),
        "n_alive": int((df["event"] == 0.0).sum()),
        "event_rate_dead": float((df["event"] == 1.0).mean()),
        "median_time_days_valid": float(df.loc[df["valid_time"], "time_to_event_days_raw"].median()),
        "median_time_years_valid": float(df.loc[df["valid_time"], "time_to_event_years_raw"].median()),
        "mean_age_years": float(df["age_years_raw"].mean()),
        "median_age_years": float(df["age_years_raw"].median()),
        "n_missing_age": int(df["missing_age"].sum()),
        "n_invalid_age": int(df["invalid_age"].sum()),
        "n_missing_sex": int(df["missing_sex"].sum()),
        "n_missing_stage": int(df["missing_stage"].sum()),
        "n_missing_treatment": int(df["missing_treatment"].sum()),
    }
    return pd.DataFrame([summary])


def build_cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cohort, g in df.groupby("cancer_type", dropna=False):
        rows.append({
            "cancer_type": cohort,
            "n_rows": int(len(g)),
            "n_unique_patients": int(g["patient_id"].nunique(dropna=True)),
            "n_duplicate_patient_rows": int(g["duplicate_patient_id"].sum()),
            "n_missing_event": int(g["missing_event"].sum()),
            "n_invalid_event": int(g["invalid_event"].sum()),
            "n_missing_time": int(g["missing_time"].sum()),
            "n_non_positive_time": int(g["non_positive_time"].sum()),
            "n_usable_for_survival_modeling": int(g["usable_for_survival_modeling"].sum()),
            "n_dead": int((g["event"] == 1.0).sum()),
            "n_alive": int((g["event"] == 0.0).sum()),
            "dead_rate": float((g["event"] == 1.0).mean()),
            "median_time_days_valid": float(g.loc[g["valid_time"], "time_to_event_days_raw"].median()),
            "median_time_years_valid": float(g.loc[g["valid_time"], "time_to_event_years_raw"].median()),
            "mean_time_days_valid": float(g.loc[g["valid_time"], "time_to_event_days_raw"].mean()),
            "mean_age_years": float(g["age_years_raw"].mean()),
            "median_age_years": float(g["age_years_raw"].median()),
            "n_missing_age": int(g["missing_age"].sum()),
            "n_invalid_age": int(g["invalid_age"].sum()),
            "n_missing_sex": int(g["missing_sex"].sum()),
            "n_missing_stage": int(g["missing_stage"].sum()),
            "n_missing_treatment": int(g["missing_treatment"].sum()),
            "omics_present_any": bool(g["omics_present"].fillna(False).astype(bool).any()) if "omics_present" in g.columns else False,
        })

    return pd.DataFrame(rows)


def build_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        rows.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": float(df[col].isna().mean() * 100.0),
            "n_non_missing": int(df[col].notna().sum()),
            "n_unique_non_missing": int(df[col].dropna().nunique()),
        })

    out = pd.DataFrame(rows).sort_values(["pct_missing", "column_name"], ascending=[False, True])
    assert len(out) == len(df.columns)
    assert n >= 0
    return out


def build_missingness_by_cohort(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for cohort, g in df.groupby("cancer_type", dropna=False):
        for col in cols:
            rows.append({
                "cancer_type": cohort,
                "column_name": col,
                "n_missing": int(g[col].isna().sum()),
                "pct_missing": float(g[col].isna().mean() * 100.0),
                "n_non_missing": int(g[col].notna().sum()),
            })
    return pd.DataFrame(rows)


def build_distribution_table(df: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    temp = df[["cancer_type", column]].copy()
    temp[column] = temp[column].fillna("").astype(str).str.strip()
    temp.loc[temp[column] == "", column] = "MISSING"

    out = (
        temp.groupby(["cancer_type", column], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cancer_type", "count", column], ascending=[True, False, True])
    )

    if top_n is not None:
        out = out.groupby("cancer_type", group_keys=False).head(top_n)

    return out


def build_eligibility_flow(df: pd.DataFrame) -> pd.DataFrame:
    n_initial = int(len(df))

    step1_mask = ~df["missing_patient_id"]
    n_after_patient_id = int(step1_mask.sum())

    step2_mask = step1_mask & ~df["missing_event"] & ~df["invalid_event"]
    n_after_event = int(step2_mask.sum())

    step3_mask = step2_mask & ~df["missing_time"]
    n_after_time_present = int(step3_mask.sum())

    step4_mask = step3_mask & ~df["non_positive_time"]
    n_final_eligible = int(step4_mask.sum())

    rows = [
        {"step": "Initial reconstructed rows", "n_rows": n_initial},
        {"step": "After excluding missing patient identifier", "n_rows": n_after_patient_id},
        {"step": "After excluding missing or invalid event", "n_rows": n_after_event},
        {"step": "After excluding missing survival time", "n_rows": n_after_time_present},
        {"step": "After excluding non-positive survival time", "n_rows": n_final_eligible},
    ]
    return pd.DataFrame(rows)


def build_time_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "time_to_event_source_raw" not in df.columns:
        return pd.DataFrame(columns=["cancer_type", "time_to_event_source_raw", "count"])

    temp = df[["cancer_type", "time_to_event_source_raw"]].copy()
    temp["time_to_event_source_raw"] = temp["time_to_event_source_raw"].fillna("").astype(str).str.strip()
    temp.loc[temp["time_to_event_source_raw"] == "", "time_to_event_source_raw"] = "MISSING"

    return (
        temp.groupby(["cancer_type", "time_to_event_source_raw"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cancer_type", "count", "time_to_event_source_raw"], ascending=[True, False, True])
    )


def build_followup_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "followup_time_source_raw" not in df.columns:
        return pd.DataFrame(columns=["cancer_type", "followup_time_source_raw", "count"])

    temp = df.loc[df["event"] == 0.0, ["cancer_type", "followup_time_source_raw"]].copy()
    temp["followup_time_source_raw"] = temp["followup_time_source_raw"].fillna("").astype(str).str.strip()
    temp.loc[temp["followup_time_source_raw"] == "", "followup_time_source_raw"] = "MISSING"

    return (
        temp.groupby(["cancer_type", "followup_time_source_raw"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cancer_type", "count", "followup_time_source_raw"], ascending=[True, False, True])
    )


def save_problem_tables(df: pd.DataFrame, audit_dir: Path) -> None:
    problem_mask = (
        df["missing_patient_id"]
        | df["duplicate_patient_id"]
        | df["missing_event"]
        | df["invalid_event"]
        | df["missing_time"]
        | df["non_positive_time"]
        | df["invalid_age"]
    )

    df.loc[problem_mask].to_csv(audit_dir / "master_table_stage1_problem_rows_detailed.csv", index=False)
    df.loc[df["duplicate_patient_id"]].to_csv(audit_dir / "duplicate_patient_rows.csv", index=False)


def make_time_histogram(df: pd.DataFrame, figures_dir: Path) -> None:
    valid = df.loc[df["valid_time"]].copy()
    if valid.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(valid["time_to_event_years_raw"], bins=40)
    plt.xlabel("Time-to-event (years, raw)")
    plt.ylabel("Number of patients")
    plt.title("Overall raw time-to-event distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "audit_time_distribution_overall.png", dpi=300)
    plt.close()


def make_time_histogram_by_cohort(df: pd.DataFrame, figures_dir: Path) -> None:
    cohorts = ordered_unique([str(x) for x in df["cancer_type"].dropna().tolist()])
    if not cohorts:
        return

    fig, axes = plt.subplots(len(cohorts), 1, figsize=(10, 4 * len(cohorts)), squeeze=False)

    for ax, cohort in zip(axes.flatten(), cohorts):
        g = df.loc[(df["cancer_type"] == cohort) & (df["valid_time"])].copy()
        if g.empty:
            ax.text(0.5, 0.5, f"No valid times for {cohort}", ha="center", va="center")
            ax.set_title(cohort)
            continue
        ax.hist(g["time_to_event_years_raw"], bins=30)
        ax.set_title(f"{cohort} raw time-to-event distribution")
        ax.set_xlabel("Time-to-event (years, raw)")
        ax.set_ylabel("Number of patients")

    plt.tight_layout()
    plt.savefig(figures_dir / "audit_time_distribution_by_cohort.png", dpi=300)
    plt.close()


def write_text_report(
    df: pd.DataFrame,
    overall_summary: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    eligibility_flow: pd.DataFrame,
    time_source_summary: pd.DataFrame,
    followup_source_summary: pd.DataFrame,
    audit_dir: Path,
) -> None:
    overall = overall_summary.iloc[0].to_dict()

    lines = []
    lines.append("TCGA MASTER TABLE STAGE 1 - DATA AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    for k, v in overall.items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("COHORT SUMMARY")
    lines.append("-" * 80)
    lines.append(cohort_summary.to_string(index=False))

    lines.append("")
    lines.append("ELIGIBILITY FLOW")
    lines.append("-" * 80)
    lines.append(eligibility_flow.to_string(index=False))

    if not time_source_summary.empty:
        lines.append("")
        lines.append("TIME-TO-EVENT SOURCE SUMMARY")
        lines.append("-" * 80)
        lines.append(time_source_summary.to_string(index=False))

    if not followup_source_summary.empty:
        lines.append("")
        lines.append("FOLLOW-UP SOURCE SUMMARY FOR CENSORED PATIENTS")
        lines.append("-" * 80)
        lines.append(followup_source_summary.to_string(index=False))

    lines.append("")
    lines.append("TOP-LEVEL AUDIT VERDICT")
    lines.append("-" * 80)

    if int(overall["n_duplicate_patient_rows"]) == 0:
        lines.append("Duplicate patient rows: PASS")
    else:
        lines.append("Duplicate patient rows: FAIL")

    if int(overall["n_invalid_event"]) == 0:
        lines.append("Invalid event labels: PASS")
    else:
        lines.append("Invalid event labels: FAIL")

    if int(overall["n_non_positive_time"]) == 0:
        lines.append("Non-positive times: PASS")
    else:
        lines.append("Non-positive times: FAIL")

    lines.append("")
    lines.append("NOTE")
    lines.append("-" * 80)
    lines.append(
        "The endpoint audit now records the source field used for final time-to-event assignment "
        "and, for censored patients, the follow-up source used to determine censoring time."
    )

    report_path = audit_dir / "master_table_stage1_audit_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    processed_dir = Path(cfg["outputs"]["processed_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])
    figures_dir = Path(cfg["outputs"]["figures_dir"])

    ensure_dir(audit_dir)
    ensure_dir(figures_dir)

    master_path = processed_dir / "master_table_stage1.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing input file: {master_path}")

    df = pd.read_csv(master_path, low_memory=False)
    validate_columns(df)
    df = add_audit_flags(df)

    overall_summary = build_overall_summary(df)
    cohort_summary = build_cohort_summary(df)
    missingness_all = build_missingness_table(df)
    eligibility_flow = build_eligibility_flow(df)
    time_source_summary = build_time_source_summary(df)
    followup_source_summary = build_followup_source_summary(df)

    important_cols = [
        "event",
        "time_to_event_days_raw",
        "age_years_raw",
        "sex_raw",
        "stage_raw",
        "received_pharmaceutical_treatment_raw",
        "race_raw",
        "ethnicity_raw",
        "omics_present",
        "time_to_event_source_raw",
        "followup_time_source_raw",
    ]
    important_cols = [c for c in important_cols if c in df.columns]
    missingness_by_cohort = build_missingness_by_cohort(df, important_cols)

    sex_dist = build_distribution_table(df, "sex_raw", top_n=None)
    stage_dist = build_distribution_table(df, "stage_raw", top_n=20)
    race_dist = build_distribution_table(df, "race_raw", top_n=20)
    ethnicity_dist = build_distribution_table(df, "ethnicity_raw", top_n=20)

    overall_summary.to_csv(audit_dir / "audit_overall_summary.csv", index=False)
    cohort_summary.to_csv(audit_dir / "audit_cohort_summary.csv", index=False)
    missingness_all.to_csv(audit_dir / "audit_missingness_all_columns.csv", index=False)
    missingness_by_cohort.to_csv(audit_dir / "audit_missingness_by_cohort.csv", index=False)
    eligibility_flow.to_csv(audit_dir / "audit_eligibility_flow.csv", index=False)
    time_source_summary.to_csv(audit_dir / "audit_time_to_event_source_summary.csv", index=False)
    followup_source_summary.to_csv(audit_dir / "audit_followup_source_summary_censored.csv", index=False)
    sex_dist.to_csv(audit_dir / "audit_sex_distribution.csv", index=False)
    stage_dist.to_csv(audit_dir / "audit_stage_distribution.csv", index=False)
    race_dist.to_csv(audit_dir / "audit_race_distribution.csv", index=False)
    ethnicity_dist.to_csv(audit_dir / "audit_ethnicity_distribution.csv", index=False)

    save_problem_tables(df, audit_dir)
    make_time_histogram(df, figures_dir)
    make_time_histogram_by_cohort(df, figures_dir)
    write_text_report(
        df,
        overall_summary,
        cohort_summary,
        eligibility_flow,
        time_source_summary,
        followup_source_summary,
        audit_dir,
    )

    json_summary = {
        "n_rows": int(len(df)),
        "n_unique_patients": int(df["patient_id"].nunique(dropna=True)),
        "n_usable_for_survival_modeling": int(df["usable_for_survival_modeling"].sum()),
        "n_missing_event": int(df["missing_event"].sum()),
        "n_invalid_event": int(df["invalid_event"].sum()),
        "n_missing_time": int(df["missing_time"].sum()),
        "n_non_positive_time": int(df["non_positive_time"].sum()),
        "eligibility_flow": eligibility_flow.to_dict(orient="records"),
        "time_to_event_source_summary": time_source_summary.to_dict(orient="records"),
        "followup_source_summary_censored": followup_source_summary.to_dict(orient="records"),
        "cohorts": cohort_summary.to_dict(orient="records"),
    }
    save_json(json_summary, audit_dir / "audit_summary.json")

    print("=" * 80)
    print("STAGE 1 DATA AUDIT COMPLETE")
    print("=" * 80)
    print(f"Input file: {master_path}")
    print(f"Overall summary saved: {audit_dir / 'audit_overall_summary.csv'}")
    print(f"Cohort summary saved: {audit_dir / 'audit_cohort_summary.csv'}")
    print(f"Eligibility flow saved: {audit_dir / 'audit_eligibility_flow.csv'}")
    print(f"Time-to-event source summary saved: {audit_dir / 'audit_time_to_event_source_summary.csv'}")
    print(f"Follow-up source summary saved: {audit_dir / 'audit_followup_source_summary_censored.csv'}")
    print(f"Missingness saved: {audit_dir / 'audit_missingness_all_columns.csv'}")
    print(f"Problem rows saved: {audit_dir / 'master_table_stage1_problem_rows_detailed.csv'}")
    print(f"Text report saved: {audit_dir / 'master_table_stage1_audit_report.txt'}")
    print(f"Figure saved: {figures_dir / 'audit_time_distribution_overall.png'}")
    print(f"Figure saved: {figures_dir / 'audit_time_distribution_by_cohort.png'}")


if __name__ == "__main__":
    main()