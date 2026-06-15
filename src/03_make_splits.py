from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


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


def validate_unique_patient_rows(df: pd.DataFrame) -> None:
    dup_mask = df.duplicated(subset=["patient_id"], keep=False)
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, "patient_id"].tolist()[:10]
        raise ValueError(
            f"Duplicate patient_id rows found in input dataset. Example IDs: {dup_ids}"
        )


def build_eligibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["cancer_type"] = out["cancer_type"].astype(str).str.strip()
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    out["time_to_event_days_raw"] = pd.to_numeric(out["time_to_event_days_raw"], errors="coerce")

    out["has_patient_id"] = out["patient_id"].ne("") & out["patient_id"].ne("nan")
    out["valid_event"] = out["event"].isin([0.0, 1.0])
    out["valid_time"] = out["time_to_event_days_raw"].notna() & (out["time_to_event_days_raw"] > 0)
    out["valid_cancer_type"] = out["cancer_type"].ne("") & out["cancer_type"].ne("nan")

    out["eligible_for_splitting"] = (
        out["has_patient_id"] & out["valid_event"] & out["valid_time"] & out["valid_cancer_type"]
    )

    return out


def make_stratify_key(df: pd.DataFrame) -> pd.Series:
    return df["cancer_type"].astype(str) + "__event" + df["event"].astype(int).astype(str)


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name, g in df.groupby("split", dropna=False):
        rows.append({
            "split": split_name,
            "cancer_type": "ALL",
            "n_patients": int(len(g)),
            "n_dead": int((g["event"] == 1).sum()),
            "n_alive": int((g["event"] == 0).sum()),
            "dead_rate": float((g["event"] == 1).mean()) if len(g) > 0 else 0.0,
            "median_time_days": float(g["time_to_event_days_raw"].median()) if len(g) > 0 else float("nan"),
        })

        for cohort, gg in g.groupby("cancer_type", dropna=False):
            rows.append({
                "split": split_name,
                "cancer_type": cohort,
                "n_patients": int(len(gg)),
                "n_dead": int((gg["event"] == 1).sum()),
                "n_alive": int((gg["event"] == 0).sum()),
                "dead_rate": float((gg["event"] == 1).mean()) if len(gg) > 0 else 0.0,
                "median_time_days": float(gg["time_to_event_days_raw"].median()) if len(gg) > 0 else float("nan"),
            })

    return pd.DataFrame(rows)


def summarize_strata(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["event"] = temp["event"].astype(int)
    out = (
        temp.groupby(["cancer_type", "event"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cancer_type", "event"])
    )
    return out


def save_json(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    processed_dir = Path(cfg["outputs"]["processed_dir"])
    splits_dir = Path(cfg["outputs"]["splits_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    ensure_dir(processed_dir)
    ensure_dir(splits_dir)
    ensure_dir(audit_dir)

    # Optional config section with safe defaults
    split_cfg = cfg.get("splits", {})
    seed = int(cfg["project"]["seed"])
    test_size = float(split_cfg.get("test_size", 0.15))
    val_size = float(split_cfg.get("val_size", 0.15))

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size <= 0 or val_size >= 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    master_path = processed_dir / "master_table_stage1.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing input file: {master_path}")

    df = pd.read_csv(master_path, low_memory=False)
    required_cols = ["patient_id", "cancer_type", "event", "time_to_event_days_raw"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = build_eligibility_flags(df)
    eligible = df.loc[df["eligible_for_splitting"]].copy()

    validate_unique_patient_rows(eligible)

    if eligible.empty:
        raise ValueError("No eligible rows available for splitting.")

    strata_table = summarize_strata(eligible)
    min_stratum = int(strata_table["count"].min())

    if min_stratum < 3:
        raise ValueError(
            "At least one cancer_type × event stratum has fewer than 3 patients, "
            "which is too small for stable train/val/test splitting."
        )

    stratify_full = make_stratify_key(eligible)

    # First split: train_val vs test
    train_val_df, test_df = train_test_split(
        eligible,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_full,
    )

    # Second split: train vs val, with validation sized relative to train_val
    val_relative = val_size / (1.0 - test_size)
    stratify_train_val = make_stratify_key(train_val_df)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=seed,
        stratify=stratify_train_val,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_assignments = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    split_assignments = split_assignments.sort_values(["split", "cancer_type", "patient_id"]).reset_index(drop=True)

    # Save eligible modeling cohort
    eligible_sorted = eligible.sort_values(["cancer_type", "patient_id"]).reset_index(drop=True)
    eligible_sorted.to_csv(processed_dir / "master_table_stage1_eligible.csv", index=False)

    # Save split assignments
    split_assignments.to_csv(splits_dir / "split_assignments.csv", index=False)
    train_df.to_csv(splits_dir / "train_split.csv", index=False)
    val_df.to_csv(splits_dir / "val_split.csv", index=False)
    test_df.to_csv(splits_dir / "test_split.csv", index=False)

    # Save simple patient-id only files too
    train_df[["patient_id"]].to_csv(splits_dir / "train_patient_ids.csv", index=False)
    val_df[["patient_id"]].to_csv(splits_dir / "val_patient_ids.csv", index=False)
    test_df[["patient_id"]].to_csv(splits_dir / "test_patient_ids.csv", index=False)

    # Save summaries
    split_summary = summarize_split(split_assignments)
    split_summary.to_csv(audit_dir / "split_summary.csv", index=False)
    strata_table.to_csv(audit_dir / "eligible_strata_counts.csv", index=False)

    # Save metadata
    metadata = {
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "train_size": 1.0 - test_size - val_size,
        "n_master_rows": int(len(df)),
        "n_eligible_rows": int(len(eligible)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "stratification": "cancer_type + event",
        "input_file": str(master_path),
        "output_files": {
            "eligible": str(processed_dir / "master_table_stage1_eligible.csv"),
            "split_assignments": str(splits_dir / "split_assignments.csv"),
            "train_split": str(splits_dir / "train_split.csv"),
            "val_split": str(splits_dir / "val_split.csv"),
            "test_split": str(splits_dir / "test_split.csv"),
            "split_summary": str(audit_dir / "split_summary.csv"),
            "eligible_strata_counts": str(audit_dir / "eligible_strata_counts.csv"),
        },
    }
    save_json(metadata, splits_dir / "split_metadata.json")

    print("=" * 80)
    print("FROZEN SPLIT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Input master table: {master_path}")
    print(f"Eligible rows: {len(eligible)} / {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Split assignments saved: {splits_dir / 'split_assignments.csv'}")
    print(f"Split summary saved: {audit_dir / 'split_summary.csv'}")
    print(f"Split metadata saved: {splits_dir / 'split_metadata.json'}")


if __name__ == "__main__":
    main()