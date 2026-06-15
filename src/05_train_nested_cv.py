from __future__ import annotations

import json
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
from sklearn.model_selection import StratifiedKFold
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


def structured_y(event: np.ndarray, time_days: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time_days.astype(float))


def cindex_from_scores(event: np.ndarray, time_days: np.ndarray, scores: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time_days.astype(float), scores)[0])


def build_param_grid(grid_spec: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def concatenate_split_arrays(a: dict[str, Any], b: dict[str, Any], matrix_key: str) -> dict[str, Any]:
    return {
        "X": np.vstack([a[matrix_key], b[matrix_key]]),
        "patient_id": a["patient_id"] + b["patient_id"],
        "cancer_type": a["cancer_type"] + b["cancer_type"],
        "event": np.concatenate([a["event"], b["event"]]),
        "time_days": np.concatenate([a["time_days"], b["time_days"]]),
    }


# -----------------------------------------------------------------------------
# Feature-set logic
# -----------------------------------------------------------------------------

def select_feature_names(feature_names: list[str], prefixes: list[str]) -> list[str]:
    return [f for f in feature_names if any(f.startswith(p) for p in prefixes)]


def build_feature_sets(feature_names: list[str]) -> dict[str, list[str]]:
    clinical_prefixes = [
        "age_years_raw",
        "received_pharmaceutical_treatment_raw",
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

    clinical_only = select_feature_names(feature_names, clinical_prefixes)
    biospecimen_only = select_feature_names(feature_names, biospecimen_prefixes)
    genomic_only = select_feature_names(feature_names, genomic_prefixes)
    full_available = list(feature_names)

    return {
        "clinical_only": clinical_only,
        "biospecimen_only": biospecimen_only,
        "genomic_only": genomic_only,
        "full_available": full_available,
    }


def subset_matrix(
    X: np.ndarray,
    all_feature_names: list[str],
    selected_feature_names: list[str],
) -> np.ndarray:
    idx = [all_feature_names.index(f) for f in selected_feature_names]
    return X[:, idx]


# -----------------------------------------------------------------------------
# Model factories
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
        scores = model.predict_partial_hazard(df_x).to_numpy().reshape(-1)
        return scores.astype(float)

    scores = np.asarray(model.predict(X)).reshape(-1)
    return scores.astype(float)


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


# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

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

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train, strat_labels), start=1):
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

                sign, train_c = orient_scores_on_training(
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

        row = {
            "model_name": model_name,
            "model_family": model_family,
            "params_json": json.dumps(params, sort_keys=True),
            "cv_mean_cindex": float(np.mean(fold_scores)) if fold_scores else np.nan,
            "cv_std_cindex": float(np.std(fold_scores)) if fold_scores else np.nan,
            "n_valid_folds": int(len(fold_scores)),
            "n_failed_folds": int(failed_folds),
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        ["cv_mean_cindex", "cv_std_cindex", "n_valid_folds"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)

    if results_df.empty or results_df["cv_mean_cindex"].isna().all():
        raise RuntimeError(f"All parameter settings failed for {model_name}.")

    best_row = results_df.iloc[0]
    best_params = json.loads(best_row["params_json"])

    return best_params, results_df


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

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

    bundle_path = processed_dir / "preprocessed_data_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing preprocessed bundle: {bundle_path}")

    bundle = joblib.load(bundle_path)

    seed = int(cfg["project"]["seed"])
    n_jobs = int(cfg["project"]["n_jobs"])

    linear_feature_names = list(bundle["linear_feature_names"])
    tree_feature_names = list(bundle["tree_feature_names"])

    feature_sets_linear = build_feature_sets(linear_feature_names)
    feature_sets_tree = build_feature_sets(tree_feature_names)

    # Save manifests
    manifest_rows = []
    for set_name, feats in feature_sets_linear.items():
        manifest_rows.append({
            "matrix_type": "linear",
            "feature_set": set_name,
            "n_features": len(feats),
            "feature_names_joined": " | ".join(feats),
        })
    for set_name, feats in feature_sets_tree.items():
        manifest_rows.append({
            "matrix_type": "tree",
            "feature_set": set_name,
            "n_features": len(feats),
            "feature_names_joined": " | ".join(feats),
        })
    pd.DataFrame(manifest_rows).to_csv(tables_dir / "feature_set_manifest.csv", index=False)

    has_genomic_features = len(feature_sets_linear["genomic_only"]) > 0
    model_registry = build_model_registry(has_genomic_features=has_genomic_features)

    # Split blocks
    tr = bundle["splits"]["train"]
    va = bundle["splits"]["val"]
    te = bundle["splits"]["test"]

    trainval_linear = concatenate_split_arrays(tr, va, "X_linear")
    trainval_tree = concatenate_split_arrays(tr, va, "X_tree")

    all_cv_rows = []
    val_rows = []
    test_rows = []
    skip_rows = []

    val_pred_rows = []
    trainval_pred_rows = []
    test_pred_rows = []

    for spec in model_registry:
        model_name = spec["model_name"]
        model_family = spec["model_family"]
        matrix_type = spec["matrix_type"]
        feature_set_name = spec["feature_set"]
        param_grid = spec["param_grid"]

        all_names = linear_feature_names if matrix_type == "linear" else tree_feature_names
        feature_set = feature_sets_linear[feature_set_name] if matrix_type == "linear" else feature_sets_tree[feature_set_name]

        if len(feature_set) == 0:
            skip_rows.append({
                "model_name": model_name,
                "reason": f"feature_set '{feature_set_name}' has zero features",
            })
            continue

        # Prepare split-specific matrices
        if matrix_type == "linear":
            X_train = subset_matrix(tr["X_linear"], linear_feature_names, feature_set)
            X_val = subset_matrix(va["X_linear"], linear_feature_names, feature_set)
            X_test = subset_matrix(te["X_linear"], linear_feature_names, feature_set)
            X_trainval = subset_matrix(trainval_linear["X"], linear_feature_names, feature_set)
        else:
            X_train = subset_matrix(tr["X_tree"], tree_feature_names, feature_set)
            X_val = subset_matrix(va["X_tree"], tree_feature_names, feature_set)
            X_test = subset_matrix(te["X_tree"], tree_feature_names, feature_set)
            X_trainval = subset_matrix(trainval_tree["X"], tree_feature_names, feature_set)

        try:
            # 1) Inner CV on train only
            best_params, cv_results = inner_cv_select(
                model_name=model_name,
                model_family=model_family,
                X_train=X_train,
                feature_names=feature_set,
                event_train=np.asarray(tr["event"], dtype=float),
                time_train=np.asarray(tr["time_days"], dtype=float),
                cancer_type_train=list(tr["cancer_type"]),
                param_grid=param_grid,
                seed=seed,
                n_jobs=n_jobs,
                n_splits=5,
            )
            cv_results["feature_set"] = feature_set_name
            cv_results["n_features"] = len(feature_set)
            all_cv_rows.append(cv_results)

            # 2) Fit chosen params on train, score on validation
            tuned_model = fit_model(
                model_family=model_family,
                params=best_params,
                X=X_train,
                feature_names=feature_set,
                event=np.asarray(tr["event"], dtype=float),
                time_days=np.asarray(tr["time_days"], dtype=float),
                random_state=seed,
                n_jobs=n_jobs,
            )

            sign_val, train_c_for_orientation = orient_scores_on_training(
                model_family=model_family,
                model=tuned_model,
                X_train=X_train,
                feature_names=feature_set,
                event_train=np.asarray(tr["event"], dtype=float),
                time_train=np.asarray(tr["time_days"], dtype=float),
            )

            val_scores = sign_val * raw_predict_scores(model_family, tuned_model, X_val, feature_set)
            val_cindex = cindex_from_scores(
                np.asarray(va["event"], dtype=float),
                np.asarray(va["time_days"], dtype=float),
                val_scores,
            )

            val_rows.append({
                "model_name": model_name,
                "model_family": model_family,
                "feature_set": feature_set_name,
                "n_features": len(feature_set),
                "best_params_json": json.dumps(best_params, sort_keys=True),
                "train_orientation_cindex": train_c_for_orientation,
                "validation_cindex": val_cindex,
            })

            for i in range(len(val_scores)):
                val_pred_rows.append({
                    "model_name": model_name,
                    "split": "val",
                    "patient_id": va["patient_id"][i],
                    "cancer_type": va["cancer_type"][i],
                    "event": float(va["event"][i]),
                    "time_days": float(va["time_days"][i]),
                    "risk_score": float(val_scores[i]),
                })

            # 3) Refit selected params on train+val, final untouched test evaluation
            final_model = fit_model(
                model_family=model_family,
                params=best_params,
                X=X_trainval,
                feature_names=feature_set,
                event=np.asarray(trainval_linear["event"] if matrix_type == "linear" else trainval_tree["event"], dtype=float),
                time_days=np.asarray(trainval_linear["time_days"] if matrix_type == "linear" else trainval_tree["time_days"], dtype=float),
                random_state=seed,
                n_jobs=n_jobs,
            )

            trainval_event = np.asarray(trainval_linear["event"] if matrix_type == "linear" else trainval_tree["event"], dtype=float)
            trainval_time = np.asarray(trainval_linear["time_days"] if matrix_type == "linear" else trainval_tree["time_days"], dtype=float)

            sign_test, trainval_c = orient_scores_on_training(
                model_family=model_family,
                model=final_model,
                X_train=X_trainval,
                feature_names=feature_set,
                event_train=trainval_event,
                time_train=trainval_time,
            )

            trainval_scores = sign_test * raw_predict_scores(model_family, final_model, X_trainval, feature_set)
            test_scores = sign_test * raw_predict_scores(model_family, final_model, X_test, feature_set)

            trainval_cindex = cindex_from_scores(trainval_event, trainval_time, trainval_scores)
            test_cindex = cindex_from_scores(
                np.asarray(te["event"], dtype=float),
                np.asarray(te["time_days"], dtype=float),
                test_scores,
            )

            test_rows.append({
                "model_name": model_name,
                "model_family": model_family,
                "feature_set": feature_set_name,
                "n_features": len(feature_set),
                "best_params_json": json.dumps(best_params, sort_keys=True),
                "trainval_cindex": trainval_cindex,
                "test_cindex": test_cindex,
                "risk_sign_applied": sign_test,
            })

            for i in range(len(trainval_scores)):
                pid = (trainval_linear["patient_id"] if matrix_type == "linear" else trainval_tree["patient_id"])[i]
                ct = (trainval_linear["cancer_type"] if matrix_type == "linear" else trainval_tree["cancer_type"])[i]
                ev = (trainval_linear["event"] if matrix_type == "linear" else trainval_tree["event"])[i]
                tm = (trainval_linear["time_days"] if matrix_type == "linear" else trainval_tree["time_days"])[i]
                trainval_pred_rows.append({
                    "model_name": model_name,
                    "split": "trainval",
                    "patient_id": pid,
                    "cancer_type": ct,
                    "event": float(ev),
                    "time_days": float(tm),
                    "risk_score": float(trainval_scores[i]),
                })

            for i in range(len(test_scores)):
                test_pred_rows.append({
                    "model_name": model_name,
                    "split": "test",
                    "patient_id": te["patient_id"][i],
                    "cancer_type": te["cancer_type"][i],
                    "event": float(te["event"][i]),
                    "time_days": float(te["time_days"][i]),
                    "risk_score": float(test_scores[i]),
                })

            # 4) Save final model artifact
            artifact = {
                "model_name": model_name,
                "model_family": model_family,
                "feature_set": feature_set_name,
                "feature_names": feature_set,
                "best_params": best_params,
                "risk_sign_applied": sign_test,
                "final_model": final_model,
            }
            joblib.dump(artifact, models_dir / f"{model_name}.joblib")

            print(f"[OK] {model_name}: val_cindex={val_cindex:.4f}, test_cindex={test_cindex:.4f}")

        except Exception as e:
            skip_rows.append({
                "model_name": model_name,
                "reason": f"{type(e).__name__}: {str(e)}",
            })
            print(f"[SKIP] {model_name}: {type(e).__name__}: {e}")

    # -----------------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------------
    if all_cv_rows:
        pd.concat(all_cv_rows, axis=0, ignore_index=True).to_csv(
            tables_dir / "model_selection_cv_results.csv", index=False
        )
    else:
        pd.DataFrame(columns=[
            "model_name", "model_family", "params_json", "cv_mean_cindex",
            "cv_std_cindex", "n_valid_folds", "n_failed_folds",
            "feature_set", "n_features"
        ]).to_csv(tables_dir / "model_selection_cv_results.csv", index=False)

    pd.DataFrame(val_rows).sort_values("validation_cindex", ascending=False).to_csv(
        tables_dir / "validation_results.csv", index=False
    )
    pd.DataFrame(test_rows).sort_values("test_cindex", ascending=False).to_csv(
        tables_dir / "final_test_results.csv", index=False
    )
    pd.DataFrame(skip_rows).to_csv(tables_dir / "skipped_models.csv", index=False)

    pd.DataFrame(val_pred_rows).to_csv(predictions_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(trainval_pred_rows).to_csv(predictions_dir / "trainval_predictions.csv", index=False)
    pd.DataFrame(test_pred_rows).to_csv(predictions_dir / "test_predictions.csv", index=False)

    metadata = {
        "bundle_path": str(bundle_path),
        "models_requested": [m["model_name"] for m in model_registry],
        "models_skipped": skip_rows,
        "seed": seed,
        "n_jobs": n_jobs,
        "feature_sets_linear": {k: len(v) for k, v in feature_sets_linear.items()},
        "feature_sets_tree": {k: len(v) for k, v in feature_sets_tree.items()},
        "outputs": {
            "cv_results": str(tables_dir / "model_selection_cv_results.csv"),
            "validation_results": str(tables_dir / "validation_results.csv"),
            "final_test_results": str(tables_dir / "final_test_results.csv"),
            "validation_predictions": str(predictions_dir / "validation_predictions.csv"),
            "trainval_predictions": str(predictions_dir / "trainval_predictions.csv"),
            "test_predictions": str(predictions_dir / "test_predictions.csv"),
        },
    }
    save_json(metadata, audit_dir / "training_metadata.json")

    print("=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"CV table: {tables_dir / 'model_selection_cv_results.csv'}")
    print(f"Validation results: {tables_dir / 'validation_results.csv'}")
    print(f"Final test results: {tables_dir / 'final_test_results.csv'}")
    print(f"Validation predictions: {predictions_dir / 'validation_predictions.csv'}")
    print(f"Train+Val predictions: {predictions_dir / 'trainval_predictions.csv'}")
    print(f"Test predictions: {predictions_dir / 'test_predictions.csv'}")
    print(f"Training metadata: {audit_dir / 'training_metadata.json'}")


if __name__ == "__main__":
    main()