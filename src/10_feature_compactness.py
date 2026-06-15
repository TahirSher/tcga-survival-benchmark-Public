from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import Surv
import warnings


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


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def fit_lifelines_cox(
    X: np.ndarray,
    feature_names: list[str],
    event: np.ndarray,
    time_days: np.ndarray,
    penalizer: float,
    l1_ratio: float,
):
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    df_fit = pd.DataFrame(X, columns=feature_names)
    df_fit["time_days"] = time_days.astype(float)
    df_fit["event"] = event.astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model.fit(df_fit, duration_col="time_days", event_col="event", show_progress=False)

    return model


def fit_rsf(
    X: np.ndarray,
    event: np.ndarray,
    time_days: np.ndarray,
    seed: int,
    n_jobs: int,
):
    model = RandomSurvivalForest(
        n_estimators=500,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(X, structured_y(event, time_days))
    return model


def fit_gb(
    X: np.ndarray,
    event: np.ndarray,
    time_days: np.ndarray,
    seed: int,
):
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X, structured_y(event, time_days))
    return model


def fit_svm(
    X: np.ndarray,
    event: np.ndarray,
    time_days: np.ndarray,
    seed: int,
):
    model = FastKernelSurvivalSVM(
        alpha=1.0,
        kernel="linear",
        max_iter=200,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X, structured_y(event, time_days))
    return model


def raw_predict_scores(model_name: str, model, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if model_name.startswith("Cox") or model_name.startswith("ElasticNetCox"):
        df_x = pd.DataFrame(X, columns=feature_names)
        return model.predict_partial_hazard(df_x).to_numpy().reshape(-1).astype(float)
    return np.asarray(model.predict(X)).reshape(-1).astype(float)


def oriented_train_test_cindex(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    train_event: np.ndarray,
    train_time: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
) -> tuple[float, float, float]:
    train_scores_raw = raw_predict_scores(model_name, model, X_train, feature_names)
    c_raw = cindex_from_scores(train_event, train_time, train_scores_raw)
    c_neg = cindex_from_scores(train_event, train_time, -train_scores_raw)

    if c_raw >= c_neg:
        sign = 1.0
        train_c = c_raw
    else:
        sign = -1.0
        train_c = c_neg

    test_scores = sign * raw_predict_scores(model_name, model, X_test, feature_names)
    test_c = cindex_from_scores(test_event, test_time, test_scores)

    return sign, train_c, test_c


def permutation_rank_features(
    model_name: str,
    final_model,
    feature_names: list[str],
    bundle: dict[str, Any],
    is_tree: bool,
    n_repeats: int = 3,
    seed: int = 42,
) -> list[str]:
    train_split = bundle["splits"]["train"]
    val_split = bundle["splits"]["val"]

    if is_tree:
        all_names = list(bundle["tree_feature_names"])
        X_train = np.vstack([train_split["X_tree"], val_split["X_tree"]])
    else:
        all_names = list(bundle["linear_feature_names"])
        X_train = np.vstack([train_split["X_linear"], val_split["X_linear"]])

    idx = [all_names.index(f) for f in feature_names]
    X_train = X_train[:, idx]

    event = np.concatenate([train_split["event"], val_split["event"]]).astype(float)
    time_days = np.concatenate([train_split["time_days"], val_split["time_days"]]).astype(float)

    raw_scores = raw_predict_scores(model_name, final_model, X_train, feature_names)
    c_raw = cindex_from_scores(event, time_days, raw_scores)
    c_neg = cindex_from_scores(event, time_days, -raw_scores)
    sign = 1.0 if c_raw >= c_neg else -1.0

    baseline_scores = sign * raw_scores
    baseline_c = cindex_from_scores(event, time_days, baseline_scores)

    rng = np.random.default_rng(seed)
    importance_rows = []

    for j, feat in enumerate(feature_names):
        drops = []

        for _ in range(n_repeats):
            X_perm = X_train.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            perm_scores = sign * raw_predict_scores(model_name, final_model, X_perm, feature_names)
            perm_c = cindex_from_scores(event, time_days, perm_scores)
            drops.append(baseline_c - perm_c)

        importance_rows.append((float(np.mean(drops)), feat))

    ranked = [feat for _, feat in sorted(importance_rows, key=lambda x: x[0], reverse=True)]
    return ranked


# -----------------------------------------------------------------------------
# Feature importance ranking
# -----------------------------------------------------------------------------

def rank_features_for_model(
    model_name: str,
    model_artifact: dict[str, Any],
    bundle: dict[str, Any],
) -> list[str]:
    feature_names = list(model_artifact["feature_names"])
    final_model = model_artifact["final_model"]

    # Tree models
    if "GradientBoostingSurvival" in model_name:
        importances = np.asarray(final_model.feature_importances_).reshape(-1)
        ranked = [f for _, f in sorted(zip(importances, feature_names), key=lambda x: x[0], reverse=True)]
        return ranked

    if "RandomSurvivalForest" in model_name:
        return permutation_rank_features(
            model_name=model_name,
            final_model=final_model,
            feature_names=feature_names,
            bundle=bundle,
            is_tree=True,
            n_repeats=3,
            seed=42,
        )

    # Cox models
    if model_name.startswith("Cox") or model_name.startswith("ElasticNetCox"):
        coefs = np.asarray(final_model.params_).reshape(-1)
        ranked = [f for _, f in sorted(zip(np.abs(coefs), feature_names), key=lambda x: x[0], reverse=True)]
        return ranked

    # SVM
    if "SurvivalSVM" in model_name:
        if hasattr(final_model, "coef_"):
            coefs = np.asarray(final_model.coef_).reshape(-1)
            ranked = [f for _, f in sorted(zip(np.abs(coefs), feature_names), key=lambda x: x[0], reverse=True)]
            return ranked

        return permutation_rank_features(
            model_name=model_name,
            final_model=final_model,
            feature_names=feature_names,
            bundle=bundle,
            is_tree=False,
            n_repeats=3,
            seed=42,
        )

    # Generic fallback
    return permutation_rank_features(
        model_name=model_name,
        final_model=final_model,
        feature_names=feature_names,
        bundle=bundle,
        is_tree=("RandomSurvivalForest" in model_name or "GradientBoostingSurvival" in model_name),
        n_repeats=3,
        seed=42,
    )


# -----------------------------------------------------------------------------
# Compactness experiment
# -----------------------------------------------------------------------------

def fit_model_by_name(
    model_name: str,
    X_train: np.ndarray,
    feature_names: list[str],
    train_event: np.ndarray,
    train_time: np.ndarray,
    seed: int,
    n_jobs: int,
):
    if model_name == "GradientBoostingSurvival_FullAvailable":
        return fit_gb(X_train, train_event, train_time, seed)
    if model_name == "RandomSurvivalForest_FullAvailable":
        return fit_rsf(X_train, train_event, train_time, seed, n_jobs)
    if model_name == "SurvivalSVM_FullAvailable":
        return fit_svm(X_train, train_event, train_time, seed)
    if model_name == "Cox_FullAvailable":
        return fit_lifelines_cox(
            X_train, feature_names, train_event, train_time,
            penalizer=0.1, l1_ratio=0.0
        )
    if model_name == "ElasticNetCox_FullAvailable":
        return fit_lifelines_cox(
            X_train, feature_names, train_event, train_time,
            penalizer=0.1, l1_ratio=0.5
        )
    raise ValueError(f"Unsupported compactness model: {model_name}")


def subset_by_feature_names(
    X: np.ndarray,
    all_names: list[str],
    selected_names: list[str],
) -> np.ndarray:
    idx = [all_names.index(f) for f in selected_names]
    return X[:, idx]


def compactness_grid(n_features_total: int) -> list[int]:
    candidates = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50]
    candidates = [k for k in candidates if k < n_features_total]
    candidates.append(n_features_total)
    return sorted(set(candidates))


def summarize_plateau(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize the compactness curve as an exploratory saturation analysis.

    Important: this summary is not intended to define a test-set-selected
    deployment model. The observed feature-count thresholds are descriptive
    summaries of the saturation curve only.
    """
    if df.empty:
        return {
            "best_test_cindex": np.nan,  # retained for backward compatibility
            "min_features_within_1pct_of_best": np.nan,  # retained for backward compatibility
            "min_features_within_95pct_of_best": np.nan,  # retained for backward compatibility
            "max_observed_test_cindex_exploratory": np.nan,
            "min_features_within_0p01_of_max_observed_test_cindex_exploratory": np.nan,
            "min_features_retaining_95pct_of_max_observed_test_cindex_exploratory": np.nan,
            "analysis_interpretation": "exploratory_saturation_analysis_not_model_selection",
        }

    max_observed = float(df["test_cindex"].max())
    threshold_0p01 = max_observed - 0.01
    threshold_95pct = max_observed * 0.95

    within_0p01 = df.loc[df["test_cindex"] >= threshold_0p01].sort_values("n_features")
    within_95 = df.loc[df["test_cindex"] >= threshold_95pct].sort_values("n_features")

    min_within_0p01 = int(within_0p01.iloc[0]["n_features"]) if not within_0p01.empty else np.nan
    min_within_95 = int(within_95.iloc[0]["n_features"]) if not within_95.empty else np.nan

    return {
        "best_test_cindex": max_observed,  # retained for backward compatibility only
        "min_features_within_1pct_of_best": min_within_0p01,  # retained for backward compatibility only
        "min_features_within_95pct_of_best": min_within_95,  # retained for backward compatibility only
        "max_observed_test_cindex_exploratory": max_observed,
        "min_features_within_0p01_of_max_observed_test_cindex_exploratory": min_within_0p01,
        "min_features_retaining_95pct_of_max_observed_test_cindex_exploratory": min_within_95,
        "analysis_interpretation": "exploratory_saturation_analysis_not_model_selection",
    }


def make_compactness_figure(compact_df: pd.DataFrame, figure_path: Path) -> None:
    if compact_df.empty:
        return

    plt.figure(figsize=(10, 6))
    for model_name, g in compact_df.groupby("model_name"):
        g = g.sort_values("n_features")
        plt.plot(g["n_features"], g["test_cindex"], marker="o", label=model_name)

    plt.xlabel("Number of top-ranked features")
    plt.ylabel("Test C-index")
    plt.title("Exploratory feature compactness / performance saturation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    seed = int(cfg["project"]["seed"])
    n_jobs = int(cfg["project"]["n_jobs"])

    processed_dir = Path(cfg["outputs"]["processed_dir"])
    models_dir = Path(cfg["outputs"]["models_dir"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    figures_dir = Path(cfg["outputs"]["figures_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    for p in [processed_dir, models_dir, tables_dir, figures_dir, audit_dir]:
        ensure_dir(p)

    bundle_path = processed_dir / "preprocessed_data_bundle.joblib"
    pooled_benchmark_path = tables_dir / "pooled_benchmark_results.csv"

    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing bundle: {bundle_path}")
    if not pooled_benchmark_path.exists():
        raise FileNotFoundError(f"Missing pooled benchmark table: {pooled_benchmark_path}")

    bundle = joblib.load(bundle_path)
    pooled_benchmark = pd.read_csv(pooled_benchmark_path)

    candidate_models = [
        "GradientBoostingSurvival_FullAvailable",
        "ElasticNetCox_FullAvailable",
        "Cox_FullAvailable",
        "RandomSurvivalForest_FullAvailable",
        "SurvivalSVM_FullAvailable",
    ]
    candidate_models = [m for m in candidate_models if m in pooled_benchmark["model_name"].tolist()]

    trainval_linear = {
        "X": np.vstack([bundle["splits"]["train"]["X_linear"], bundle["splits"]["val"]["X_linear"]]),
        "event": np.concatenate([bundle["splits"]["train"]["event"], bundle["splits"]["val"]["event"]]),
        "time_days": np.concatenate([bundle["splits"]["train"]["time_days"], bundle["splits"]["val"]["time_days"]]),
    }
    trainval_tree = {
        "X": np.vstack([bundle["splits"]["train"]["X_tree"], bundle["splits"]["val"]["X_tree"]]),
        "event": np.concatenate([bundle["splits"]["train"]["event"], bundle["splits"]["val"]["event"]]),
        "time_days": np.concatenate([bundle["splits"]["train"]["time_days"], bundle["splits"]["val"]["time_days"]]),
    }
    test_linear = {
        "X": bundle["splits"]["test"]["X_linear"],
        "event": np.asarray(bundle["splits"]["test"]["event"], dtype=float),
        "time_days": np.asarray(bundle["splits"]["test"]["time_days"], dtype=float),
    }
    test_tree = {
        "X": bundle["splits"]["test"]["X_tree"],
        "event": np.asarray(bundle["splits"]["test"]["event"], dtype=float),
        "time_days": np.asarray(bundle["splits"]["test"]["time_days"], dtype=float),
    }

    all_rows = []
    plateau_rows = []
    ranking_rows = []
    skipped_rows = []

    for model_name in candidate_models:
        artifact_path = models_dir / f"{model_name}.joblib"
        if not artifact_path.exists():
            skipped_rows.append({"model_name": model_name, "reason": "missing final model artifact"})
            continue

        artifact = joblib.load(artifact_path)
        ranked_features = rank_features_for_model(model_name, artifact, bundle)
        if len(ranked_features) == 0:
            skipped_rows.append({"model_name": model_name, "reason": "no ranked features"})
            continue

        for rank_idx, feat in enumerate(ranked_features, start=1):
            ranking_rows.append({
                "model_name": model_name,
                "rank": rank_idx,
                "feature_name": feat,
                "analysis_interpretation": "exploratory_saturation_analysis_not_model_selection",
            })

        is_tree = ("RandomSurvivalForest" in model_name) or ("GradientBoostingSurvival" in model_name)
        all_names = list(bundle["tree_feature_names"]) if is_tree else list(bundle["linear_feature_names"])
        train_block = trainval_tree if is_tree else trainval_linear
        test_block = test_tree if is_tree else test_linear

        grid = compactness_grid(len(ranked_features))

        for k in grid:
            selected = ranked_features[:k]

            try:
                X_train = subset_by_feature_names(train_block["X"], all_names, selected)
                X_test = subset_by_feature_names(test_block["X"], all_names, selected)

                model = fit_model_by_name(
                    model_name=model_name,
                    X_train=X_train,
                    feature_names=selected,
                    train_event=np.asarray(train_block["event"], dtype=float),
                    train_time=np.asarray(train_block["time_days"], dtype=float),
                    seed=seed,
                    n_jobs=n_jobs,
                )

                _, train_c, test_c = oriented_train_test_cindex(
                    model_name=model_name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=selected,
                    train_event=np.asarray(train_block["event"], dtype=float),
                    train_time=np.asarray(train_block["time_days"], dtype=float),
                    test_event=np.asarray(test_block["event"], dtype=float),
                    test_time=np.asarray(test_block["time_days"], dtype=float),
                )

                all_rows.append({
                    "model_name": model_name,
                    "n_features": int(k),
                    "trainval_cindex": float(train_c),
                    "test_cindex": float(test_c),
                    "selected_features_joined": " | ".join(selected),
                    "analysis_interpretation": "exploratory_saturation_analysis_not_model_selection",
                })

                print(f"[OK] {model_name} | top-{k}: test_cindex={test_c:.4f}")

            except Exception as e:
                skipped_rows.append({
                    "model_name": model_name,
                    "n_features": int(k),
                    "reason": f"{type(e).__name__}: {str(e)}",
                })
                print(f"[SKIP] {model_name} | top-{k}: {type(e).__name__}: {e}")

    compact_df = pd.DataFrame(all_rows)
    ranking_df = pd.DataFrame(ranking_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if not compact_df.empty:
        for model_name, g in compact_df.groupby("model_name"):
            plateau = summarize_plateau(g.sort_values("n_features"))
            plateau_rows.append({
                "model_name": model_name,
                **plateau,
            })

    plateau_df = pd.DataFrame(plateau_rows)

    compactness_path = tables_dir / "feature_compactness_results.csv"
    ranking_path = tables_dir / "feature_rankings_for_compactness.csv"
    plateau_path = tables_dir / "feature_compactness_plateau_summary.csv"
    skipped_path = tables_dir / "feature_compactness_skipped.csv"
    figure_path = figures_dir / "feature_compactness_curve.png"

    compact_df.to_csv(compactness_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    plateau_df.to_csv(plateau_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)
    make_compactness_figure(compact_df, figure_path)

    summary = {
        "candidate_models": candidate_models,
        "analysis_interpretation": "exploratory_saturation_analysis_not_model_selection",
        "interpretation_note": (
            "Feature compactness was treated as an exploratory saturation analysis. "
            "The full curve was reported to show robustness of performance across "
            "feature counts and was not intended as a test-set-driven model-selection procedure."
        ),
        "n_compactness_rows": int(len(compact_df)),
        "n_ranking_rows": int(len(ranking_df)),
        "n_skipped_rows": int(len(skipped_df)),
        "outputs": {
            "compactness_results": str(compactness_path),
            "feature_rankings": str(ranking_path),
            "plateau_summary": str(plateau_path),
            "skipped": str(skipped_path),
            "figure": str(figure_path),
        },
    }
    save_json(summary, audit_dir / "feature_compactness_summary.json")

    print("=" * 80)
    print("FEATURE COMPACTNESS ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Compactness table: {compactness_path}")
    print(f"Feature ranking table: {ranking_path}")
    print(f"Plateau summary table: {plateau_path}")
    print(f"Skipped table: {skipped_path}")
    print(f"Figure: {figure_path}")
    print(f"Summary: {audit_dir / 'feature_compactness_summary.json'}")


if __name__ == "__main__":
    main()