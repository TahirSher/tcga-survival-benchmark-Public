from __future__ import annotations

import json
import re
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import Surv


CONFIG_PATH = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project\configs\config.yaml")


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def structured_y(event: np.ndarray, time_days: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time_days.astype(float))


def cindex_from_scores(event: np.ndarray, time_days: np.ndarray, scores: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time_days.astype(float), scores)[0])


def build_param_grid(grid_spec: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


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

    out["sex_clean"] = out["sex_raw"].apply(clean_sex) if "sex_raw" in out.columns else "unknown"
    out["race_clean"] = out["race_raw"].apply(clean_simple_category) if "race_raw" in out.columns else "missing"
    out["ethnicity_clean"] = out["ethnicity_raw"].apply(clean_simple_category) if "ethnicity_raw" in out.columns else "missing"
    out["primary_site_clean"] = out["primary_site_raw"].apply(clean_simple_category) if "primary_site_raw" in out.columns else "missing"
    out["stage_group"] = out["stage_raw"].apply(stage_group) if "stage_raw" in out.columns else "missing"

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

    return out


def build_feature_lists(df_train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    core_numeric = [
        "age_years_raw",
        # "received_pharmaceutical_treatment_raw",  # excluded from primary feature sets
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
        "received_pharmaceutical_treatment_raw",
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
        "usable_for_survival_modeling",
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


def get_feature_names_after_variance(pipeline: Pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocessor"]
    var = pipeline.named_steps["variance"]
    base_names = pre.get_feature_names_out()
    keep_mask = var.get_support()
    return [name for name, keep in zip(base_names, keep_mask) if keep]


def select_feature_names(feature_names: list[str], prefixes: list[str]) -> list[str]:
    return [f for f in feature_names if any(f.startswith(p) for p in prefixes)]


def build_feature_sets(feature_names: list[str]) -> dict[str, list[str]]:
    clinical_prefixes = [
        "age_years_raw",
        # "received_pharmaceutical_treatment_raw",  # excluded from primary feature sets
        "cancer_type_",
        "sex_clean_",
        "race_clean_",
        "ethnicity_clean_",
        "primary_site_clean_",
        "stage_group_",
    ]

    biospecimen_prefixes = [
        "biospecimen_n_sample_slots_non_null",
        "biospecimen_n_ffpe_true",
        "bio_has_",
        "omics_present",
        "omics_n_columns",
    ]

    genomic_prefixes = [
        "rs_",
        "gene_",
        "mut_",
        "mutation_",
        "snv_",
        "maf_",
        "variant_",
        "cnv_",
        "methyl_",
        "expr_",
        "expression_",
        "fpkm_",
        "htseq_",
    ]

    return {
        "clinical_only": select_feature_names(feature_names, clinical_prefixes),
        "biospecimen_only": select_feature_names(feature_names, biospecimen_prefixes),
        "genomic_only": select_feature_names(feature_names, genomic_prefixes),
        "full_available": list(feature_names),
    }


def subset_matrix(
    X: np.ndarray,
    all_feature_names: list[str],
    selected_feature_names: list[str],
) -> np.ndarray:
    idx = [all_feature_names.index(f) for f in selected_feature_names]
    return X[:, idx]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def make_model(model_family: str, params: dict[str, Any], random_state: int, n_jobs: int):
    if model_family == "lifelines_cox":
        return CoxPHFitter(
            penalizer=float(params.get("penalizer", 0.0)),
            l1_ratio=float(params.get("l1_ratio", 0.0)),
        )

    if model_family == "rsf":
        return RandomSurvivalForest(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if model_family == "gb":
        return GradientBoostingSurvivalAnalysis(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            subsample=float(params["subsample"]),
            random_state=random_state,
        )

    if model_family == "svm":
        return FastKernelSurvivalSVM(
            alpha=float(params["alpha"]),
            kernel=params["kernel"],
            max_iter=int(params["max_iter"]),
            random_state=random_state,
        )

    raise ValueError(f"Unknown model family: {model_family}")


def fit_model(
    model_family: str,
    params: dict[str, Any],
    X: np.ndarray,
    feature_names: list[str],
    event: np.ndarray,
    time_days: np.ndarray,
    random_state: int,
    n_jobs: int,
):
    model = make_model(model_family, params, random_state=random_state, n_jobs=n_jobs)

    if model_family == "lifelines_cox":
        df_fit = pd.DataFrame(X, columns=feature_names)
        df_fit["time_days"] = time_days.astype(float)
        df_fit["event"] = event.astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model.fit(df_fit, duration_col="time_days", event_col="event", show_progress=False)
    else:
        y = structured_y(event, time_days)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X, y)

    return model


def raw_predict_scores(
    model_family: str,
    model,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    if model_family == "lifelines_cox":
        df_x = pd.DataFrame(X, columns=feature_names)
        return model.predict_partial_hazard(df_x).to_numpy().reshape(-1).astype(float)
    return np.asarray(model.predict(X)).reshape(-1).astype(float)


def orient_scores_on_training(
    model_family: str,
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    event_train: np.ndarray,
    time_train: np.ndarray,
) -> tuple[float, float]:
    raw_scores = raw_predict_scores(model_family, model, X_train, feature_names)
    c_raw = cindex_from_scores(event_train, time_train, raw_scores)
    c_neg = cindex_from_scores(event_train, time_train, -raw_scores)
    if c_raw >= c_neg:
        return 1.0, c_raw
    return -1.0, c_neg


def inner_cv_select(
    model_name: str,
    model_family: str,
    X_train: np.ndarray,
    feature_names: list[str],
    event_train: np.ndarray,
    time_train: np.ndarray,
    cancer_type_train: list[str],
    param_grid: list[dict[str, Any]],
    seed: int,
    n_jobs: int,
    n_splits: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    strat_labels = np.array(
        [f"{ct}__event{int(ev)}" for ct, ev in zip(cancer_type_train, event_train)]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []

    for params in param_grid:
        fold_scores = []
        failed_folds = 0

        for tr_idx, va_idx in cv.split(X_train, strat_labels):
            X_tr = X_train[tr_idx]
            X_va = X_train[va_idx]
            ev_tr = event_train[tr_idx]
            ev_va = event_train[va_idx]
            tm_tr = time_train[tr_idx]
            tm_va = time_train[va_idx]

            try:
                model = fit_model(
                    model_family=model_family,
                    params=params,
                    X=X_tr,
                    feature_names=feature_names,
                    event=ev_tr,
                    time_days=tm_tr,
                    random_state=seed,
                    n_jobs=n_jobs,
                )

                sign, _ = orient_scores_on_training(
                    model_family=model_family,
                    model=model,
                    X_train=X_tr,
                    feature_names=feature_names,
                    event_train=ev_tr,
                    time_train=tm_tr,
                )

                va_scores = sign * raw_predict_scores(model_family, model, X_va, feature_names)
                va_c = cindex_from_scores(ev_va, tm_va, va_scores)
                fold_scores.append(va_c)
            except Exception:
                failed_folds += 1

        rows.append({
            "model_name": model_name,
            "model_family": model_family,
            "params_json": json.dumps(params, sort_keys=True),
            "cv_mean_cindex": float(np.mean(fold_scores)) if fold_scores else np.nan,
            "cv_std_cindex": float(np.std(fold_scores)) if fold_scores else np.nan,
            "n_valid_folds": int(len(fold_scores)),
            "n_failed_folds": int(failed_folds),
        })

    results_df = pd.DataFrame(rows).sort_values(
        ["cv_mean_cindex", "cv_std_cindex", "n_valid_folds"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)

    if results_df.empty or results_df["cv_mean_cindex"].isna().all():
        raise RuntimeError(f"All parameter settings failed for {model_name}.")

    best_params = json.loads(results_df.iloc[0]["params_json"])
    return best_params, results_df


def build_model_registry(has_genomic_features: bool) -> list[dict[str, Any]]:
    registry = [
        {
            "model_name": "Cox_ClinicalOnly",
            "model_family": "lifelines_cox",
            "matrix_type": "linear",
            "feature_set": "clinical_only",
            "param_grid": build_param_grid({
                "penalizer": [0.01, 0.1, 0.5],
                "l1_ratio": [0.0],
            }),
        },
        {
            "model_name": "Cox_FullAvailable",
            "model_family": "lifelines_cox",
            "matrix_type": "linear",
            "feature_set": "full_available",
            "param_grid": build_param_grid({
                "penalizer": [0.01, 0.1, 0.5],
                "l1_ratio": [0.0],
            }),
        },
        {
            "model_name": "ElasticNetCox_FullAvailable",
            "model_family": "lifelines_cox",
            "matrix_type": "linear",
            "feature_set": "full_available",
            "param_grid": build_param_grid({
                "penalizer": [0.01, 0.1, 0.5, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8],
            }),
        },
        {
            "model_name": "RandomSurvivalForest_FullAvailable",
            "model_family": "rsf",
            "matrix_type": "tree",
            "feature_set": "full_available",
            "param_grid": build_param_grid({
                "n_estimators": [300, 500],
                "max_depth": [5, None],
                "min_samples_split": [10, 20],
                "min_samples_leaf": [5, 10],
                "max_features": ["sqrt"],
            }),
        },
        {
            "model_name": "GradientBoostingSurvival_FullAvailable",
            "model_family": "gb",
            "matrix_type": "tree",
            "feature_set": "full_available",
            "param_grid": build_param_grid({
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "max_depth": [1, 2],
                "min_samples_split": [10, 20],
                "min_samples_leaf": [5, 10],
                "subsample": [0.8],
            }),
        },
        {
            "model_name": "SurvivalSVM_FullAvailable",
            "model_family": "svm",
            "matrix_type": "linear",
            "feature_set": "full_available",
            "param_grid": build_param_grid({
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "max_iter": [200],
            }),
        },
    ]

    if has_genomic_features:
        registry.insert(
            1,
            {
                "model_name": "Cox_GenomicOnly",
                "model_family": "lifelines_cox",
                "matrix_type": "linear",
                "feature_set": "genomic_only",
                "param_grid": build_param_grid({
                    "penalizer": [0.01, 0.1, 0.5],
                    "l1_ratio": [0.0],
                }),
            },
        )

    return registry


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    processed_dir = Path(cfg["outputs"]["processed_dir"])
    models_dir = Path(cfg["outputs"]["models_dir"])
    predictions_dir = Path(cfg["outputs"]["predictions_dir"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    for p in [processed_dir, models_dir, predictions_dir, tables_dir, audit_dir]:
        ensure_dir(p)

    eligible_path = processed_dir / "master_table_stage1_eligible.csv"
    if not eligible_path.exists():
        raise FileNotFoundError(f"Missing eligible master table: {eligible_path}")

    df = pd.read_csv(eligible_path, low_memory=False)
    df = engineer_features(df)

    required_cols = ["patient_id", "cancer_type", "event", "time_to_event_days_raw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in eligible table: {missing}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["cancer_type"] = df["cancer_type"].astype(str).str.strip()
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["time_to_event_days_raw"] = pd.to_numeric(df["time_to_event_days_raw"], errors="coerce")
    df = df.dropna(subset=["patient_id", "cancer_type", "event", "time_to_event_days_raw"]).copy()
    df = df.loc[df["time_to_event_days_raw"] > 0].copy()

    if df.duplicated(subset=["patient_id"]).any():
        dup_ids = df.loc[df.duplicated(subset=["patient_id"], keep=False), "patient_id"].tolist()[:10]
        raise ValueError(f"Duplicate patient_id rows remain in eligible table. Example IDs: {dup_ids}")

    seed = int(cfg["project"]["seed"])
    n_jobs = int(cfg["project"]["n_jobs"])
    holdouts = sorted(df["cancer_type"].dropna().unique().tolist())

    all_results = []
    all_cv_results = []
    all_predictions = []
    feature_manifest_rows = []
    skipped_rows = []

    for holdout_cancer in holdouts:
        train_df = df.loc[df["cancer_type"] != holdout_cancer].copy()
        test_df = df.loc[df["cancer_type"] == holdout_cancer].copy()

        if train_df.empty or test_df.empty:
            skipped_rows.append({
                "holdout_cancer": holdout_cancer,
                "model_name": "ALL",
                "reason": "empty train or test fold",
            })
            continue

        numeric_features, categorical_features, extra_numeric = build_feature_lists(train_df)
        linear_pipeline, tree_pipeline = build_preprocessors(numeric_features, categorical_features)

        linear_pipeline.fit(train_df)
        tree_pipeline.fit(train_df)

        X_train_linear = linear_pipeline.transform(train_df)
        X_test_linear = linear_pipeline.transform(test_df)
        X_train_tree = tree_pipeline.transform(train_df)
        X_test_tree = tree_pipeline.transform(test_df)

        linear_feature_names = get_feature_names_after_variance(linear_pipeline)
        tree_feature_names = get_feature_names_after_variance(tree_pipeline)

        feature_sets_linear = build_feature_sets(linear_feature_names)
        feature_sets_tree = build_feature_sets(tree_feature_names)

        for set_name, feats in feature_sets_linear.items():
            feature_manifest_rows.append({
                "holdout_cancer": holdout_cancer,
                "matrix_type": "linear",
                "feature_set": set_name,
                "n_features": len(feats),
                "feature_names_joined": " | ".join(feats),
            })
        for set_name, feats in feature_sets_tree.items():
            feature_manifest_rows.append({
                "holdout_cancer": holdout_cancer,
                "matrix_type": "tree",
                "feature_set": set_name,
                "n_features": len(feats),
                "feature_names_joined": " | ".join(feats),
            })

        has_genomic_features = len(feature_sets_linear["genomic_only"]) > 0
        registry = build_model_registry(has_genomic_features=has_genomic_features)

        train_event = train_df["event"].to_numpy(dtype=float)
        train_time = train_df["time_to_event_days_raw"].to_numpy(dtype=float)
        test_event = test_df["event"].to_numpy(dtype=float)
        test_time = test_df["time_to_event_days_raw"].to_numpy(dtype=float)

        for spec in registry:
            model_name = spec["model_name"]
            model_family = spec["model_family"]
            matrix_type = spec["matrix_type"]
            feature_set_name = spec["feature_set"]
            param_grid = spec["param_grid"]

            all_names = linear_feature_names if matrix_type == "linear" else tree_feature_names
            feature_set = feature_sets_linear[feature_set_name] if matrix_type == "linear" else feature_sets_tree[feature_set_name]

            if len(feature_set) == 0:
                skipped_rows.append({
                    "holdout_cancer": holdout_cancer,
                    "model_name": model_name,
                    "reason": f"feature_set '{feature_set_name}' has zero features",
                })
                continue

            if matrix_type == "linear":
                X_train = subset_matrix(X_train_linear, linear_feature_names, feature_set)
                X_test = subset_matrix(X_test_linear, linear_feature_names, feature_set)
            else:
                X_train = subset_matrix(X_train_tree, tree_feature_names, feature_set)
                X_test = subset_matrix(X_test_tree, tree_feature_names, feature_set)

            try:
                best_params, cv_results = inner_cv_select(
                    model_name=model_name,
                    model_family=model_family,
                    X_train=X_train,
                    feature_names=feature_set,
                    event_train=train_event,
                    time_train=train_time,
                    cancer_type_train=train_df["cancer_type"].tolist(),
                    param_grid=param_grid,
                    seed=seed,
                    n_jobs=n_jobs,
                    n_splits=5,
                )
                cv_results["holdout_cancer"] = holdout_cancer
                cv_results["feature_set"] = feature_set_name
                cv_results["n_features"] = len(feature_set)
                all_cv_results.append(cv_results)

                final_model = fit_model(
                    model_family=model_family,
                    params=best_params,
                    X=X_train,
                    feature_names=feature_set,
                    event=train_event,
                    time_days=train_time,
                    random_state=seed,
                    n_jobs=n_jobs,
                )

                sign, train_cindex = orient_scores_on_training(
                    model_family=model_family,
                    model=final_model,
                    X_train=X_train,
                    feature_names=feature_set,
                    event_train=train_event,
                    time_train=train_time,
                )

                test_scores = sign * raw_predict_scores(model_family, final_model, X_test, feature_set)
                test_cindex = cindex_from_scores(test_event, test_time, test_scores)

                all_results.append({
                    "holdout_cancer": holdout_cancer,
                    "model_name": model_name,
                    "model_family": model_family,
                    "feature_set": feature_set_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_features": len(feature_set),
                    "train_cindex": train_cindex,
                    "holdout_test_cindex": test_cindex,
                    "best_params_json": json.dumps(best_params, sort_keys=True),
                    "risk_sign_applied": sign,
                })

                for i in range(len(test_scores)):
                    all_predictions.append({
                        "holdout_cancer": holdout_cancer,
                        "model_name": model_name,
                        "patient_id": test_df["patient_id"].iloc[i],
                        "cancer_type": test_df["cancer_type"].iloc[i],
                        "event": float(test_event[i]),
                        "time_days": float(test_time[i]),
                        "risk_score": float(test_scores[i]),
                    })

                artifact = {
                    "holdout_cancer": holdout_cancer,
                    "model_name": model_name,
                    "model_family": model_family,
                    "feature_set": feature_set_name,
                    "feature_names": feature_set,
                    "best_params": best_params,
                    "risk_sign_applied": sign,
                    "linear_feature_names": linear_feature_names if matrix_type == "linear" else None,
                    "tree_feature_names": tree_feature_names if matrix_type == "tree" else None,
                    "preprocessor_pipeline": linear_pipeline if matrix_type == "linear" else tree_pipeline,
                    "final_model": final_model,
                }
                joblib.dump(
                    artifact,
                    models_dir / f"LOCO_{holdout_cancer}_{model_name}.joblib",
                )

                print(f"[OK] holdout={holdout_cancer} | {model_name}: cindex={test_cindex:.4f}")

            except Exception as e:
                skipped_rows.append({
                    "holdout_cancer": holdout_cancer,
                    "model_name": model_name,
                    "reason": f"{type(e).__name__}: {str(e)}",
                })
                print(f"[SKIP] holdout={holdout_cancer} | {model_name}: {type(e).__name__}: {e}")

    results_df = pd.DataFrame(all_results)
    cv_df = pd.concat(all_cv_results, axis=0, ignore_index=True) if all_cv_results else pd.DataFrame()
    preds_df = pd.DataFrame(all_predictions)
    feature_manifest_df = pd.DataFrame(feature_manifest_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if not results_df.empty:
        results_df = results_df.sort_values(["holdout_cancer", "holdout_test_cindex"], ascending=[True, False]).reset_index(drop=True)

    results_path = tables_dir / "leave_one_cancer_out_results.csv"
    cv_path = tables_dir / "leave_one_cancer_out_cv_results.csv"
    preds_path = predictions_dir / "leave_one_cancer_out_predictions.csv"
    manifest_path = tables_dir / "leave_one_cancer_out_feature_manifest.csv"
    skipped_path = tables_dir / "leave_one_cancer_out_skipped_models.csv"

    results_df.to_csv(results_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    feature_manifest_df.to_csv(manifest_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    summary = {
        "holdout_cancers": holdouts,
        "n_results_rows": int(len(results_df)),
        "n_prediction_rows": int(len(preds_df)),
        "n_skipped_rows": int(len(skipped_df)),
        "outputs": {
            "results": str(results_path),
            "cv_results": str(cv_path),
            "predictions": str(preds_path),
            "feature_manifest": str(manifest_path),
            "skipped_models": str(skipped_path),
        },
    }
    save_json(summary, audit_dir / "leave_one_cancer_out_summary.json")

    print("=" * 80)
    print("LEAVE-ONE-CANCER-OUT EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results table: {results_path}")
    print(f"CV table: {cv_path}")
    print(f"Predictions table: {preds_path}")
    print(f"Feature manifest: {manifest_path}")
    print(f"Skipped models: {skipped_path}")
    print(f"Summary: {audit_dir / 'leave_one_cancer_out_summary.json'}")


if __name__ == "__main__":
    main()