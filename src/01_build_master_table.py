from __future__ import annotations

import ast
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "[]", "{}"}


def first_non_missing(row: pd.Series, candidates: list[str]) -> Any:
    for col in candidates:
        if col in row.index and not is_missing(row[col]):
            return row[col]
    return np.nan


def to_numeric(value: Any) -> float:
    if is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# -----------------------------------------------------------------------------
# Nested parsing helpers
# -----------------------------------------------------------------------------

def parse_nested_string(value: Any) -> Any:
    """
    Safely parse strings that may contain JSON-like or Python-literal content.
    Returns the parsed object when possible, otherwise returns the original value.
    """
    if is_missing(value):
        return None

    if isinstance(value, (dict, list, tuple)):
        return value

    text = str(value).strip()
    if text in {"[]", "{}", ""}:
        return None

    # Try strict JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try Python literal form next
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def recursive_key_search(obj: Any, key_patterns: list[str]) -> list[Any]:
    """
    Recursively search nested dict/list structures for values whose keys
    match any regex in key_patterns.
    """
    results: list[Any] = []
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in key_patterns]

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                key_text = str(k)
                if any(p.search(key_text) for p in compiled):
                    results.append(v)
                _walk(v)
        elif isinstance(x, list):
            for item in x:
                _walk(item)

    _walk(obj)
    return results


def flatten_nested_text(value: Any) -> str:
    parsed = parse_nested_string(value)
    if parsed is None:
        return ""

    chunks: list[str] = []

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                chunks.append(str(k))
                _walk(v)
        elif isinstance(x, list):
            for item in x:
                _walk(item)
        else:
            if not is_missing(x):
                chunks.append(str(x))

    _walk(parsed)
    return " | ".join(chunks)


# -----------------------------------------------------------------------------
# Column inspection / audit
# -----------------------------------------------------------------------------

def save_column_inventory(df: pd.DataFrame, out_csv: Path, source_name: str) -> None:
    inv = pd.DataFrame({
        "source": source_name,
        "column_name": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "null": [int(df[c].isna().sum()) for c in df.columns],
    })
    inv.to_csv(out_csv, index=False)


def find_columns_by_regex(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    compiled = [re.compile(p, flags=re.IGNORECASE) for p in patterns]
    return [c for c in df.columns if any(p.search(c) for p in compiled)]


# -----------------------------------------------------------------------------
# Clinical extraction
# -----------------------------------------------------------------------------

def extract_event(vital_status: Any) -> float:
    text = normalize_text(vital_status).lower()
    if text == "":
        return np.nan
    if "dead" in text:
        return 1.0
    if "alive" in text:
        return 0.0
    return np.nan


def extract_age_years(row: pd.Series, age_candidates: list[str]) -> float:
    # Prefer already-usable age field
    for col in age_candidates:
        if col in row.index and not is_missing(row[col]):
            val = to_numeric(row[col])
            if np.isnan(val):
                continue

            # days_to_birth is usually negative days
            if "days_to_birth" in col.lower():
                if val == 0:
                    return np.nan
                return abs(val) / 365.25

            # age_at_index is typically years
            return val
    return np.nan


def extract_followup_days(row: pd.Series) -> tuple[float, str]:
    """
    Deterministically assign censoring time using ordered precedence.

    Ordered precedence:
    1. days_to_last_follow_up
    2. days_to_last_known_disease_status
    3. days_to_last_followup
    4. days_to_last_known_alive
    5. days_to_last_contact
    6. other cohort-specific follow-up style fields

    If multiple positive values are present within the same precedence level,
    the maximum positive value is retained.
    """
    precedence = [
        ("days_to_last_follow_up", [r"^days_to_last_follow_up$"]),
        ("days_to_last_known_disease_status", [r"^days_to_last_known_disease_status$"]),
        ("days_to_last_followup", [r"^days_to_last_followup$"]),
        ("days_to_last_known_alive", [r"^days_to_last_known_alive$"]),
        ("days_to_last_contact", [r"^days_to_last_contact$"]),
        ("other_followup_field", [r"days_to_.*follow", r"last_follow", r"follow_up", r"followup"]),
    ]

    nested_candidates = ["follow_ups", "diagnoses", "exposures"]

    for source_name, patterns in precedence:
        found: list[float] = []

        # direct columns
        direct_cols = [c for c in row.index if any(re.search(p, c, flags=re.IGNORECASE) for p in patterns)]
        for col in direct_cols:
            if not is_missing(row[col]):
                num = to_numeric(row[col])
                if not np.isnan(num) and num > 0:
                    found.append(num)

        # nested columns
        for col in nested_candidates:
            if col in row.index and not is_missing(row[col]):
                parsed = parse_nested_string(row[col])
                values = recursive_key_search(parsed, patterns)
                for v in values:
                    num = to_numeric(v)
                    if not np.isnan(num) and num > 0:
                        found.append(num)

        if found:
            return float(max(found)), source_name

    return np.nan, ""


def extract_stage(row: pd.Series) -> str:
    stage_patterns = [
        r"ajcc.*stage",
        r"pathologic_stage",
        r"clinical_stage",
        r"tumor_stage",
        r"stage_event",
    ]

    # direct columns first
    direct_cols = find_columns_by_regex(pd.DataFrame([row]), stage_patterns)
    for col in direct_cols:
        text = normalize_text(row[col])
        if text:
            return text

    # nested text from diagnoses/follow_ups
    for col in ["diagnoses", "follow_ups"]:
        if col in row.index and not is_missing(row[col]):
            parsed = parse_nested_string(row[col])
            values = recursive_key_search(parsed, stage_patterns)
            values = [normalize_text(v) for v in values if normalize_text(v)]
            if values:
                return " | ".join(pd.unique(values))

    return ""


def extract_treatment_flag(row: pd.Series) -> float:
    text_parts = []
    for col in ["treatments", "diagnoses", "exposures", "follow_ups"]:
        if col in row.index and not is_missing(row[col]):
            text_parts.append(flatten_nested_text(row[col]).lower())

    if not text_parts:
        return np.nan

    text = " ".join(text_parts)

    positive_keywords = [
        "pharmaceutical",
        "drug",
        "chemotherapy",
        "systemic therapy",
        "systemic treatment",
        "targeted therapy",
        "treatment_type",
        "therapy_type",
        "treatment_or_therapy",
    ]
    negative_keywords = [
        "no treatment",
        "not administered",
        "none",
    ]

    if any(k in text for k in positive_keywords):
        return 1.0
    if any(k in text for k in negative_keywords):
        return 0.0

    return np.nan


def extract_primary_site(row: pd.Series) -> str:
    for col in ["primary_site", "disease_type", "project.project_id", "project_project_id"]:
        if col in row.index and not is_missing(row[col]):
            return normalize_text(row[col])
    return ""


def build_clinical_patient_table(
    df: pd.DataFrame,
    cohort_name: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    bcfg = cfg["build_master_table"]

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        patient_id = first_non_missing(row, bcfg["patient_id_priority"])
        case_id = row["case_id"] if "case_id" in row.index else np.nan

        vital_status = first_non_missing(row, bcfg["event_candidates"])
        event = extract_event(vital_status)

        days_to_death = first_non_missing(row, bcfg["survival_time_candidates"])
        days_to_death = to_numeric(days_to_death)

        followup_days, followup_source = extract_followup_days(row)

        if not np.isnan(event):
            if event == 1.0:
                time_to_event_days = days_to_death if not np.isnan(days_to_death) and days_to_death > 0 else np.nan
                time_to_event_source = "days_to_death" if not np.isnan(time_to_event_days) else ""
            else:
                time_to_event_days = followup_days if not np.isnan(followup_days) and followup_days > 0 else np.nan
                time_to_event_source = followup_source if not np.isnan(time_to_event_days) else ""
        else:
            time_to_event_days = np.nan
            time_to_event_source = ""

        age_years = extract_age_years(row, bcfg["age_candidates"])
        sex = first_non_missing(row, bcfg["sex_candidates"])
        primary_site = extract_primary_site(row)
        stage = extract_stage(row)
        treatment_flag = extract_treatment_flag(row)

        records.append({
            "patient_id": normalize_text(patient_id),
            "case_id": normalize_text(case_id),
            "cancer_type": cohort_name,
            "primary_site_raw": primary_site,
            "vital_status_raw": normalize_text(vital_status),
            "event": event,
            "days_to_death_raw": days_to_death,
            "days_to_last_followup_raw": followup_days,
            "followup_time_source_raw": followup_source,
            "time_to_event_days_raw": time_to_event_days,
            "time_to_event_source_raw": time_to_event_source,
            "age_years_raw": age_years,
            "sex_raw": normalize_text(sex),
            "stage_raw": stage,
            "received_pharmaceutical_treatment_raw": treatment_flag,
            "race_raw": normalize_text(row["demographic.race"]) if "demographic.race" in row.index else "",
            "ethnicity_raw": normalize_text(row["demographic.ethnicity"]) if "demographic.ethnicity" in row.index else "",
            "clinical_source_has_followups": float("follow_ups" in row.index),
            "clinical_source_has_diagnoses": float("diagnoses" in row.index),
        })

    out = pd.DataFrame(records)

    # Patient-level collapse in case of duplicate rows
    agg_dict = {
        "case_id": "first",
        "cancer_type": "first",
        "primary_site_raw": "first",
        "vital_status_raw": "first",
        "event": "max",
        "days_to_death_raw": "max",
        "days_to_last_followup_raw": "max",
        "followup_time_source_raw": lambda s: next((x for x in s if normalize_text(x)), ""),
        "time_to_event_days_raw": "max",
        "time_to_event_source_raw": lambda s: next((x for x in s if normalize_text(x)), ""),
        "age_years_raw": "first",
        "sex_raw": "first",
        "stage_raw": lambda s: next((x for x in s if normalize_text(x)), ""),
        "received_pharmaceutical_treatment_raw": "max",
        "race_raw": "first",
        "ethnicity_raw": "first",
        "clinical_source_has_followups": "max",
        "clinical_source_has_diagnoses": "max",
    }

    out = (
        out.loc[out["patient_id"].astype(str).str.len() > 0]
        .groupby("patient_id", as_index=False)
        .agg(agg_dict)
    )

    return out


# -----------------------------------------------------------------------------
# Biospecimen extraction
# -----------------------------------------------------------------------------

def summarize_unique_values(row: pd.Series, columns: list[str]) -> str:
    values = []
    for col in columns:
        if col in row.index and not is_missing(row[col]):
            values.append(normalize_text(row[col]))
    values = [v for v in values if v]
    if not values:
        return ""
    return " | ".join(pd.unique(values))


def count_truthy_ffpe(row: pd.Series, columns: list[str]) -> int:
    count = 0
    for col in columns:
        if col in row.index and not is_missing(row[col]):
            text = normalize_text(row[col]).lower()
            if text in {"true", "1", "yes"}:
                count += 1
    return count


def build_biospecimen_patient_table(df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    sample_type_cols = find_columns_by_regex(df, [r"samples_\d+_sample_type$"])
    specimen_type_cols = find_columns_by_regex(df, [r"samples_\d+_specimen_type$"])
    tissue_type_cols = find_columns_by_regex(df, [r"samples_\d+_tissue_type$"])
    tumor_descriptor_cols = find_columns_by_regex(df, [r"samples_\d+_tumor_descriptor$"])
    ffpe_cols = find_columns_by_regex(df, [r"is_ffpe$"])

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        patient_id = normalize_text(first_non_missing(row, ["submitter_id", "case_id"]))

        n_sample_type_non_null = int(sum(not is_missing(row[c]) for c in sample_type_cols if c in row.index))

        records.append({
            "patient_id": patient_id,
            "cancer_type": cohort_name,
            "biospecimen_case_id": normalize_text(row["case_id"]) if "case_id" in row.index else "",
            "biospecimen_sample_types": summarize_unique_values(row, sample_type_cols),
            "biospecimen_specimen_types": summarize_unique_values(row, specimen_type_cols),
            "biospecimen_tissue_types": summarize_unique_values(row, tissue_type_cols),
            "biospecimen_tumor_descriptors": summarize_unique_values(row, tumor_descriptor_cols),
            "biospecimen_n_sample_slots_non_null": n_sample_type_non_null,
            "biospecimen_n_ffpe_true": count_truthy_ffpe(row, ffpe_cols),
        })

    out = pd.DataFrame(records)

    out = (
        out.loc[out["patient_id"].astype(str).str.len() > 0]
        .groupby("patient_id", as_index=False)
        .agg({
            "cancer_type": "first",
            "biospecimen_case_id": "first",
            "biospecimen_sample_types": lambda s: next((x for x in s if normalize_text(x)), ""),
            "biospecimen_specimen_types": lambda s: next((x for x in s if normalize_text(x)), ""),
            "biospecimen_tissue_types": lambda s: next((x for x in s if normalize_text(x)), ""),
            "biospecimen_tumor_descriptors": lambda s: next((x for x in s if normalize_text(x)), ""),
            "biospecimen_n_sample_slots_non_null": "max",
            "biospecimen_n_ffpe_true": "max",
        })
    )

    return out


# -----------------------------------------------------------------------------
# Omics audit only
# -----------------------------------------------------------------------------

def inspect_omics_zip(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists():
        return {
            "omics_present": False,
            "omics_file": "",
            "omics_rows_preview": 0,
            "omics_n_columns": 0,
            "omics_first_columns": "",
        }

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if not members:
            return {
                "omics_present": True,
                "omics_file": "",
                "omics_rows_preview": 0,
                "omics_n_columns": 0,
                "omics_first_columns": "",
            }

        member = members[0]
        with zf.open(member) as f:
            sample = pd.read_csv(f, nrows=3)

    return {
        "omics_present": True,
        "omics_file": member,
        "omics_rows_preview": int(sample.shape[0]),
        "omics_n_columns": int(sample.shape[1]),
        "omics_first_columns": " | ".join(map(str, sample.columns[:25])),
    }


# -----------------------------------------------------------------------------
# Main build logic
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    audit_dir = Path(cfg["outputs"]["audit_dir"])
    interim_dir = Path(cfg["outputs"]["interim_dir"])
    processed_dir = Path(cfg["outputs"]["processed_dir"])
    logs_dir = Path(cfg["outputs"]["logs_dir"])

    for p in [audit_dir, interim_dir, processed_dir, logs_dir]:
        ensure_dir(p)

    combined_tables: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []

    for cohort_name, cohort_cfg in cfg["cohorts"].items():
        cohort_dir = Path(cohort_cfg["directory"])
        print(f"\n{'=' * 80}")
        print(f"Processing cohort: {cohort_name}")
        print(f"{'=' * 80}")

        clinical_csv_path = cohort_dir / cohort_cfg["clinical_csv"]
        biospecimen_csv_path = cohort_dir / cohort_cfg["biospecimen_csv"]

        clinical_df = pd.read_csv(clinical_csv_path, low_memory=False)
        biospecimen_df = pd.read_csv(biospecimen_csv_path, low_memory=False)

        save_column_inventory(
            clinical_df,
            audit_dir / f"{cohort_name.lower()}_clinical_columns.csv",
            f"{cohort_name}_clinical",
        )
        save_column_inventory(
            biospecimen_df,
            audit_dir / f"{cohort_name.lower()}_biospecimen_columns.csv",
            f"{cohort_name}_biospecimen",
        )

        clinical_pt = build_clinical_patient_table(clinical_df, cohort_name, cfg)
        biospecimen_pt = build_biospecimen_patient_table(biospecimen_df, cohort_name)

        merged = clinical_pt.merge(
            biospecimen_pt.drop(columns=["cancer_type"], errors="ignore"),
            how="left",
            on="patient_id",
        )

        omics_zip_name = cohort_cfg.get("omics_zip")
        omics_info = {
            "omics_present": False,
            "omics_file": "",
            "omics_rows_preview": 0,
            "omics_n_columns": 0,
            "omics_first_columns": "",
        }
        if omics_zip_name:
            omics_info = inspect_omics_zip(cohort_dir / omics_zip_name)

        for k, v in omics_info.items():
            merged[k] = v

        merged["has_valid_event"] = merged["event"].notna().astype(int)
        merged["has_valid_time"] = merged["time_to_event_days_raw"].notna().astype(int)
        merged["time_positive_flag"] = (merged["time_to_event_days_raw"] > 0).fillna(False).astype(int)

        merged.to_csv(interim_dir / f"{cohort_name.lower()}_patient_master_stage1.csv", index=False)

        summary_records.append({
            "cancer_type": cohort_name,
            "n_clinical_rows": int(clinical_df.shape[0]),
            "n_biospecimen_rows": int(biospecimen_df.shape[0]),
            "n_patient_rows_after_merge": int(merged.shape[0]),
            "n_event_non_missing": int(merged["event"].notna().sum()),
            "n_time_non_missing": int(merged["time_to_event_days_raw"].notna().sum()),
            "n_positive_time": int((merged["time_to_event_days_raw"] > 0).fillna(False).sum()),
            "n_dead": int((merged["event"] == 1).sum()),
            "n_alive": int((merged["event"] == 0).sum()),
            "omics_present": bool(omics_info["omics_present"]),
            "omics_n_columns": int(omics_info["omics_n_columns"]),
        })

        combined_tables.append(merged)

        print(f"Clinical rows: {clinical_df.shape[0]}")
        print(f"Biospecimen rows: {biospecimen_df.shape[0]}")
        print(f"Patient-level merged rows: {merged.shape[0]}")
        print(f"Non-missing event: {merged['event'].notna().sum()}")
        print(f"Non-missing time: {merged['time_to_event_days_raw'].notna().sum()}")
        print(f"Positive time rows: {(merged['time_to_event_days_raw'] > 0).fillna(False).sum()}")

    master = pd.concat(combined_tables, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)

    master.to_csv(processed_dir / "master_table_stage1.csv", index=False)
    summary_df.to_csv(audit_dir / "master_table_stage1_summary.csv", index=False)

    json_summary = {
        "project_name": cfg["project"]["name"],
        "seed": cfg["project"]["seed"],
        "n_jobs": cfg["project"]["n_jobs"],
        "n_total_rows": int(master.shape[0]),
        "n_total_columns": int(master.shape[1]),
        "cohort_summary": summary_records,
    }

    with (audit_dir / "master_table_stage1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("MASTER TABLE STAGE 1 BUILD COMPLETE")
    print(f"{'=' * 80}")
    print(f"Saved: {processed_dir / 'master_table_stage1.csv'}")
    print(f"Saved: {audit_dir / 'master_table_stage1_summary.csv'}")
    print(f"Saved: {audit_dir / 'master_table_stage1_summary.json'}")


if __name__ == "__main__":
    main()