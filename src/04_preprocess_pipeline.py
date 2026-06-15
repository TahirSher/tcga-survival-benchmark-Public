from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def save_json(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def clean_sex(x: Any) -> str:
    text = normalize_text(x).lower()
    if text in {"female", "f"}:
        return "female"
    if text in {"male", "m"}:
        return "male"
    return "unknown"


def clean_simple_category(x: Any) -> str:
    text = normalize_text(x).lower()
    return text if text else "missing"


def stage_group(x: Any) -> str:
    text = normalize_text(x).lower()
    if text == "":
        return "missing"

    if re.search(r"\bstage\s*iv\b|\biv[a-c]?\b", text):
        return "stage_iv"
    if re.search(r"\bstage\s*iii\b|\biii[a-c]?\b", text):
        return "stage_iii"
    if re.search(r"\bstage\s*ii\b|\bii[a-c]?\b", text):
        return "stage_ii"
    if re.search(r"\bstage\s*i\b|\bi[a-c]?\b", text):
        return "stage_i"
    if re.search(r"\bstage\s*0\b|\b0\b", text):
        return "stage_0"
    if re.search(r"\bstage\s*x\b|\bx\b", text):
        return "stage_x"
    return "stage_other"


def has_keyword(text: Any, keywords: list[str]) -> int:
    t = normalize_text(text).lower()
    return int(any(k in t for k in keywords))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Normalize base text columns
    for col in [
        "primary_site_raw",
        "sex_raw",
        "stage_raw",
        "race_raw",
        "ethnicity_raw",
        "biospecimen_sample_types",
        "biospecimen_specimen_types",
        "biospecimen_tissue_types",
        "biospecimen_tumor_descriptors",
    ]:
        if col in out.columns:
            out[col] = out[col].apply(normalize_text)

    # Clean numeric columns
    for col in [
        "event",
        "time_to_event_days_raw",
        "age_years_raw",
        "received_pharmaceutical_treatment_raw",
        "biospecimen_n_sample_slots_non_null",
        "biospecimen_n_ffpe_true",
        "omics_n_columns",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "omics_present" in out.columns:
        out["omics_present"] = out["omics_present"].fillna(False).astype(bool).astype(int)

    # Derived compact categorical variables
    out["sex_clean"] = out["sex_raw"].apply(clean_sex) if "sex_raw" in out.columns else "unknown"
    out["race_clean"] = out["race_raw"].apply(clean_simple_category) if "race_raw" in out.columns else "missing"
    out["ethnicity_clean"] = out["ethnicity_raw"].apply(clean_simple_category) if "ethnicity_raw" in out.columns else "missing"
    out["primary_site_clean"] = out["primary_site_raw"].apply(clean_simple_category) if "primary_site_raw" in out.columns else "missing"
    out["stage_group"] = out["stage_raw"].apply(stage_group) if "stage_raw" in out.columns else "missing"

    # Derived biospecimen flags
    biospecimen_text = (
        out.get("biospecimen_sample_types", "").astype(str) + " | " +
        out.get("biospecimen_specimen_types", "").astype(str) + " | " +
        out.get("biospecimen_tissue_types", "").astype(str) + " | " +
        out.get("biospecimen_tumor_descriptors", "").astype(str)
    )

    out["bio_has_primary_tumor"] = biospecimen_text.apply(lambda x: has_keyword(x, ["primary tumor", "primary"]))
    out["bio_has_normal_sample"] = biospecimen_text.apply(lambda x: has_keyword(x, ["normal"]))
    out["bio_has_blood_normal"] = biospecimen_text.apply(lambda x: has_keyword(x, ["blood derived normal", "peripheral blood"]))
    out["bio_has_solid_tissue_normal"] = biospecimen_text.apply(lambda x: has_keyword(x, ["solid tissue normal"]))
    out["bio_has_ffpe_flag"] = (
        pd.to_numeric(out.get("biospecimen_n_ffpe_true", 0), errors="coerce").fillna(0) > 0
    ).astype(int)

    # Optional future-proof numeric auto-detection
    exclude_for_auto_numeric = {
        "event",
        "time_to_event_days_raw",
        "days_to_death_raw",
        "days_to_last_followup_raw",
        "received_pharmaceutical_treatment_raw",  # excluded from primary feature sets
        "clinical_source_has_followups",
        "clinical_source_has_diagnoses",
        "omics_rows_preview",
        "has_valid_event",
        "has_valid_time",
        "time_positive_flag",
    }

    auto_numeric = []
    for col in out.columns:
        if col in exclude_for_auto_numeric:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            auto_numeric.append(col)

    out.attrs["auto_numeric_detected"] = auto_numeric
    return out


# -----------------------------------------------------------------------------
# Feature selection for preprocessing
# -----------------------------------------------------------------------------

def build_feature_lists(df_train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    core_numeric = [
        "age_years_raw",
        # "received_pharmaceutical_treatment_raw",  # removed from primary feature sets
        "biospecimen_n_sample_slots_non_null",
        "biospecimen_n_ffpe_true",
        "omics_present",
        "omics_n_columns",
        "bio_has_primary_tumor",
        "bio_has_normal_sample",
        "bio_has_blood_normal",
        "bio_has_solid_tissue_normal",
        "bio_has_ffpe_flag",
    ]

    core_categorical = [
        "cancer_type",
        "sex_clean",
        "race_clean",
        "ethnicity_clean",
        "primary_site_clean",
        "stage_group",
    ]

    meta_exclude = {
        "patient_id",
        "case_id",
        "split",
        "event",
        "time_to_event_days_raw",
        "days_to_death_raw",
        "days_to_last_followup_raw",
        "vital_status_raw",
        "stage_raw",
        "sex_raw",
        "race_raw",
        "ethnicity_raw",
        "primary_site_raw",
        "biospecimen_case_id",
        "biospecimen_sample_types",
        "biospecimen_specimen_types",
        "biospecimen_tissue_types",
        "biospecimen_tumor_descriptors",
        "omics_file",
        "omics_rows_preview",
        "omics_first_columns",
        "received_pharmaceutical_treatment_raw",  # excluded from primary feature sets
        "clinical_source_has_followups",
        "clinical_source_has_diagnoses",
        "has_valid_event",
        "has_valid_time",
        "time_positive_flag",
        "has_patient_id",
        "valid_event",
        "valid_time",
        "valid_cancer_type",
        "eligible_for_splitting",
    }

    present_core_numeric = [c for c in core_numeric if c in df_train.columns]
    present_core_categorical = [c for c in core_categorical if c in df_train.columns]

    extra_numeric = []
    for col in df_train.columns:
        if col in meta_exclude or col in present_core_numeric or col in present_core_categorical:
            continue
        if pd.api.types.is_numeric_dtype(df_train[col]):
            extra_numeric.append(col)

    numeric_features = present_core_numeric + sorted(extra_numeric)
    categorical_features = present_core_categorical

    return numeric_features, categorical_features, extra_numeric


def build_preprocessors(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, Pipeline]:
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    linear_pipeline = Pipeline([
        ("preprocessor", linear_preprocessor),
        ("variance", VarianceThreshold(threshold=0.0)),
    ])

    tree_pipeline = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("variance", VarianceThreshold(threshold=0.0)),
    ])

    return linear_pipeline, tree_pipeline


def transform_split(
    pipeline: Pipeline,
    df: pd.DataFrame,
    patient_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    X = pipeline.transform(df)
    meta = df[patient_cols].copy().reset_index(drop=True)
    return X, meta


def get_feature_names_after_variance(pipeline: Pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocessor"]
    var = pipeline.named_steps["variance"]

    base_names = pre.get_feature_names_out()
    keep_mask = var.get_support()
    return [name for name, keep in zip(base_names, keep_mask) if keep]


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

    train_path = splits_dir / "train_split.csv"
    val_path = splits_dir / "val_split.csv"
    test_path = splits_dir / "test_split.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    numeric_features, categorical_features, extra_numeric = build_feature_lists(train_df)

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise ValueError("No usable features found for preprocessing.")

    # Train-only missingness summary before imputation
    train_missingness_rows = []
    for col in numeric_features + categorical_features:
        train_missingness_rows.append({
            "feature": col,
            "feature_type": "numeric" if col in numeric_features else "categorical",
            "n_missing_train": int(train_df[col].isna().sum()) if col in train_df.columns else None,
            "pct_missing_train": float(train_df[col].isna().mean() * 100.0) if col in train_df.columns else None,
        })
    pd.DataFrame(train_missingness_rows).to_csv(
        audit_dir / "preprocessing_train_missingness.csv", index=False
    )

    linear_pipeline, tree_pipeline = build_preprocessors(numeric_features, categorical_features)

    # Fit only on train
    linear_pipeline.fit(train_df)
    tree_pipeline.fit(train_df)

    patient_cols = ["patient_id", "cancer_type", "event", "time_to_event_days_raw"]

    X_train_linear, meta_train = transform_split(linear_pipeline, train_df, patient_cols)
    X_val_linear, meta_val = transform_split(linear_pipeline, val_df, patient_cols)
    X_test_linear, meta_test = transform_split(linear_pipeline, test_df, patient_cols)

    X_train_tree, _ = transform_split(tree_pipeline, train_df, patient_cols)
    X_val_tree, _ = transform_split(tree_pipeline, val_df, patient_cols)
    X_test_tree, _ = transform_split(tree_pipeline, test_df, patient_cols)

    linear_feature_names = get_feature_names_after_variance(linear_pipeline)
    tree_feature_names = get_feature_names_after_variance(tree_pipeline)

    # Save feature manifests
    pd.DataFrame({"feature_name": linear_feature_names}).to_csv(
        processed_dir / "linear_feature_names.csv", index=False
    )
    pd.DataFrame({"feature_name": tree_feature_names}).to_csv(
        processed_dir / "tree_feature_names.csv", index=False
    )
    pd.DataFrame({"numeric_feature": numeric_features}).to_csv(
        processed_dir / "selected_numeric_features.csv", index=False
    )
    pd.DataFrame({"categorical_feature": categorical_features}).to_csv(
        processed_dir / "selected_categorical_features.csv", index=False
    )
    pd.DataFrame({"auto_extra_numeric_feature": extra_numeric}).to_csv(
        processed_dir / "auto_extra_numeric_features.csv", index=False
    )

    # Save split target/meta manifests
    meta_train.to_csv(processed_dir / "train_targets_and_ids.csv", index=False)
    meta_val.to_csv(processed_dir / "val_targets_and_ids.csv", index=False)
    meta_test.to_csv(processed_dir / "test_targets_and_ids.csv", index=False)

    # Save joblib bundle
    bundle = {
        "config_seed": int(cfg["project"]["seed"]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "extra_numeric_features": extra_numeric,
        "linear_feature_names": linear_feature_names,
        "tree_feature_names": tree_feature_names,
        "linear_pipeline": linear_pipeline,
        "tree_pipeline": tree_pipeline,
        "splits": {
            "train": {
                "X_linear": X_train_linear,
                "X_tree": X_train_tree,
                "patient_id": meta_train["patient_id"].tolist(),
                "cancer_type": meta_train["cancer_type"].tolist(),
                "event": meta_train["event"].astype(float).to_numpy(),
                "time_days": meta_train["time_to_event_days_raw"].astype(float).to_numpy(),
            },
            "val": {
                "X_linear": X_val_linear,
                "X_tree": X_val_tree,
                "patient_id": meta_val["patient_id"].tolist(),
                "cancer_type": meta_val["cancer_type"].tolist(),
                "event": meta_val["event"].astype(float).to_numpy(),
                "time_days": meta_val["time_to_event_days_raw"].astype(float).to_numpy(),
            },
            "test": {
                "X_linear": X_test_linear,
                "X_tree": X_test_tree,
                "patient_id": meta_test["patient_id"].tolist(),
                "cancer_type": meta_test["cancer_type"].tolist(),
                "event": meta_test["event"].astype(float).to_numpy(),
                "time_days": meta_test["time_to_event_days_raw"].astype(float).to_numpy(),
            },
        },
    }
    joblib.dump(bundle, processed_dir / "preprocessed_data_bundle.joblib")

    # Save preprocessing summary
    summary = {
        "n_train": int(X_train_linear.shape[0]),
        "n_val": int(X_val_linear.shape[0]),
        "n_test": int(X_test_linear.shape[0]),
        "n_numeric_features_before_encoding": int(len(numeric_features)),
        "n_categorical_features_before_encoding": int(len(categorical_features)),
        "n_extra_numeric_auto_detected": int(len(extra_numeric)),
        "n_linear_features_after_encoding_and_variance_filter": int(len(linear_feature_names)),
        "n_tree_features_after_encoding_and_variance_filter": int(len(tree_feature_names)),
        "train_linear_shape": [int(x) for x in X_train_linear.shape],
        "val_linear_shape": [int(x) for x in X_val_linear.shape],
        "test_linear_shape": [int(x) for x in X_test_linear.shape],
        "train_tree_shape": [int(x) for x in X_train_tree.shape],
        "val_tree_shape": [int(x) for x in X_val_tree.shape],
        "test_tree_shape": [int(x) for x in X_test_tree.shape],
        "output_bundle": str(processed_dir / "preprocessed_data_bundle.joblib"),
    }
    save_json(summary, audit_dir / "preprocessing_summary.json")

    print("=" * 80)
    print("LEAKAGE-SAFE PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Train rows: {X_train_linear.shape[0]}")
    print(f"Validation rows: {X_val_linear.shape[0]}")
    print(f"Test rows: {X_test_linear.shape[0]}")
    print(f"Numeric features before encoding: {len(numeric_features)}")
    print(f"Categorical features before encoding: {len(categorical_features)}")
    print(f"Extra numeric auto-detected: {len(extra_numeric)}")
    print(f"Linear feature count after preprocessing: {len(linear_feature_names)}")
    print(f"Tree feature count after preprocessing: {len(tree_feature_names)}")
    print(f"Bundle saved: {processed_dir / 'preprocessed_data_bundle.joblib'}")
    print(f"Summary saved: {audit_dir / 'preprocessing_summary.json'}")


if __name__ == "__main__":
    main()