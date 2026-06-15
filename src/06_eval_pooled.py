from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
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


def compute_cindex(event: np.ndarray, time_days: np.ndarray, risk_score: np.ndarray) -> float:
    event = np.asarray(event).astype(bool)
    time_days = np.asarray(time_days).astype(float)
    risk_score = np.asarray(risk_score).astype(float)
    return float(concordance_index_censored(event, time_days, risk_score)[0])


def structured_y(event: np.ndarray, time_days: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time_days).astype(float))


def compute_uno_cindex(
    train_event_ref: np.ndarray,
    train_time_ref: np.ndarray,
    test_event: np.ndarray,
    test_time: np.ndarray,
    test_risk_score: np.ndarray,
) -> dict[str, float]:
    train_event_ref = np.asarray(train_event_ref).astype(bool)
    train_time_ref = np.asarray(train_time_ref).astype(float)
    test_event = np.asarray(test_event).astype(bool)
    test_time = np.asarray(test_time).astype(float)
    test_risk_score = np.asarray(test_risk_score).astype(float)

    if len(train_event_ref) == 0 or len(test_event) == 0:
        return {
            "uno_cindex": np.nan,
            "uno_tau_days": np.nan,
            "n_train_ref_rows": int(len(train_event_ref)),
            "n_test_rows_used": 0,
        }

    train_event_times = train_time_ref[train_event_ref]
    if len(train_event_times) == 0:
        return {
            "uno_cindex": np.nan,
            "uno_tau_days": np.nan,
            "n_train_ref_rows": int(len(train_event_ref)),
            "n_test_rows_used": 0,
        }

    tau = float(min(np.max(train_event_times), np.max(test_time)))
    if not np.isfinite(tau) or tau <= 0:
        return {
            "uno_cindex": np.nan,
            "uno_tau_days": np.nan,
            "n_train_ref_rows": int(len(train_event_ref)),
            "n_test_rows_used": 0,
        }

    test_mask = test_time <= tau
    if int(test_mask.sum()) == 0:
        return {
            "uno_cindex": np.nan,
            "uno_tau_days": tau,
            "n_train_ref_rows": int(len(train_event_ref)),
            "n_test_rows_used": 0,
        }

    try:
        y_train_ref = structured_y(train_event_ref, train_time_ref)
        y_test = structured_y(test_event[test_mask], test_time[test_mask])
        uno_value = float(
            concordance_index_ipcw(
                y_train_ref,
                y_test,
                test_risk_score[test_mask],
                tau=tau,
            )[0]
        )
    except Exception:
        uno_value = np.nan

    return {
        "uno_cindex": uno_value,
        "uno_tau_days": tau,
        "n_train_ref_rows": int(len(train_event_ref)),
        "n_test_rows_used": int(test_mask.sum()),
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["cancer_type"] = out["cancer_type"].astype(str).str.strip()
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    out["time_days"] = pd.to_numeric(out["time_days"], errors="coerce")
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out = out.dropna(subset=["patient_id", "cancer_type", "event", "time_days", "risk_score"]).copy()
    return out


# -----------------------------------------------------------------------------
# Reference survival data for Uno's C-index
# -----------------------------------------------------------------------------

def build_trainval_reference(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for split_name in ["train", "val"]:
        split = bundle["splits"][split_name]
        n = len(split["patient_id"])
        for i in range(n):
            rows.append({
                "patient_id": str(split["patient_id"][i]).strip(),
                "cancer_type": str(split["cancer_type"][i]).strip(),
                "event": float(split["event"][i]),
                "time_days": float(split["time_days"][i]),
            })

    ref_df = pd.DataFrame(rows)
    ref_df["event"] = pd.to_numeric(ref_df["event"], errors="coerce")
    ref_df["time_days"] = pd.to_numeric(ref_df["time_days"], errors="coerce")
    ref_df = ref_df.dropna(subset=["cancer_type", "event", "time_days"]).copy()
    return ref_df


# -----------------------------------------------------------------------------
# Bootstrap helpers
# -----------------------------------------------------------------------------

def bootstrap_cindex(
    pred_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(pred_df)
    values = []

    if n == 0:
        return {
            "bootstrap_mean_cindex": np.nan,
            "bootstrap_std_cindex": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "n_bootstrap_valid": 0,
        }

    event = pred_df["event"].to_numpy(dtype=float)
    time_days = pred_df["time_days"].to_numpy(dtype=float)
    risk_score = pred_df["risk_score"].to_numpy(dtype=float)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c = compute_cindex(event[idx], time_days[idx], risk_score[idx])
            if np.isfinite(c):
                values.append(c)
        except Exception:
            continue

    if len(values) == 0:
        return {
            "bootstrap_mean_cindex": np.nan,
            "bootstrap_std_cindex": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "n_bootstrap_valid": 0,
        }

    values = np.asarray(values, dtype=float)
    return {
        "bootstrap_mean_cindex": float(np.mean(values)),
        "bootstrap_std_cindex": float(np.std(values)),
        "ci_lower_95": float(np.percentile(values, 2.5)),
        "ci_upper_95": float(np.percentile(values, 97.5)),
        "n_bootstrap_valid": int(len(values)),
    }


def paired_bootstrap_delta_vs_baseline(
    baseline_df: pd.DataFrame,
    model_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    merged = baseline_df[["patient_id", "event", "time_days", "risk_score"]].rename(
        columns={"risk_score": "risk_score_baseline"}
    ).merge(
        model_df[["patient_id", "risk_score"]].rename(columns={"risk_score": "risk_score_model"}),
        on="patient_id",
        how="inner",
    )

    if merged.empty:
        return {
            "delta_cindex_point": np.nan,
            "delta_ci_lower_95": np.nan,
            "delta_ci_upper_95": np.nan,
            "delta_bootstrap_mean": np.nan,
            "delta_bootstrap_std": np.nan,
            "n_bootstrap_valid": 0,
            "n_paired_rows": 0,
        }

    event = merged["event"].to_numpy(dtype=float)
    time_days = merged["time_days"].to_numpy(dtype=float)
    score_base = merged["risk_score_baseline"].to_numpy(dtype=float)
    score_model = merged["risk_score_model"].to_numpy(dtype=float)

    point_base = compute_cindex(event, time_days, score_base)
    point_model = compute_cindex(event, time_days, score_model)
    point_delta = point_model - point_base

    rng = np.random.default_rng(seed)
    n = len(merged)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c_base = compute_cindex(event[idx], time_days[idx], score_base[idx])
            c_model = compute_cindex(event[idx], time_days[idx], score_model[idx])
            delta = c_model - c_base
            if np.isfinite(delta):
                deltas.append(delta)
        except Exception:
            continue

    if len(deltas) == 0:
        return {
            "delta_cindex_point": float(point_delta),
            "delta_ci_lower_95": np.nan,
            "delta_ci_upper_95": np.nan,
            "delta_bootstrap_mean": np.nan,
            "delta_bootstrap_std": np.nan,
            "n_bootstrap_valid": 0,
            "n_paired_rows": int(len(merged)),
        }

    deltas = np.asarray(deltas, dtype=float)
    return {
        "delta_cindex_point": float(point_delta),
        "delta_ci_lower_95": float(np.percentile(deltas, 2.5)),
        "delta_ci_upper_95": float(np.percentile(deltas, 97.5)),
        "delta_bootstrap_mean": float(np.mean(deltas)),
        "delta_bootstrap_std": float(np.std(deltas)),
        "n_bootstrap_valid": int(len(deltas)),
        "n_paired_rows": int(len(merged)),
    }


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def build_per_model_bootstrap_table(
    test_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for model_name, g in test_predictions.groupby("model_name", dropna=False):
        point_c = compute_cindex(
            g["event"].to_numpy(dtype=float),
            g["time_days"].to_numpy(dtype=float),
            g["risk_score"].to_numpy(dtype=float),
        )
        boot = bootstrap_cindex(g, n_bootstrap=n_bootstrap, seed=seed)
        rows.append({
            "model_name": model_name,
            "test_cindex_from_predictions": point_c,
            **boot,
        })
    return pd.DataFrame(rows).sort_values("test_cindex_from_predictions", ascending=False)


def build_per_model_uno_table(
    test_predictions: pd.DataFrame,
    trainval_ref_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    train_event_ref = trainval_ref_df["event"].to_numpy(dtype=float)
    train_time_ref = trainval_ref_df["time_days"].to_numpy(dtype=float)

    for model_name, g in test_predictions.groupby("model_name", dropna=False):
        uno = compute_uno_cindex(
            train_event_ref=train_event_ref,
            train_time_ref=train_time_ref,
            test_event=g["event"].to_numpy(dtype=float),
            test_time=g["time_days"].to_numpy(dtype=float),
            test_risk_score=g["risk_score"].to_numpy(dtype=float),
        )
        rows.append({
            "model_name": model_name,
            **uno,
        })

    return pd.DataFrame(rows).sort_values("uno_cindex", ascending=False, na_position="last")


def build_per_cohort_test_table(
    test_predictions: pd.DataFrame,
    trainval_ref_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (model_name, cancer_type), g in test_predictions.groupby(["model_name", "cancer_type"], dropna=False):
        ref_g = trainval_ref_df.loc[trainval_ref_df["cancer_type"] == cancer_type].copy()

        uno = compute_uno_cindex(
            train_event_ref=ref_g["event"].to_numpy(dtype=float),
            train_time_ref=ref_g["time_days"].to_numpy(dtype=float),
            test_event=g["event"].to_numpy(dtype=float),
            test_time=g["time_days"].to_numpy(dtype=float),
            test_risk_score=g["risk_score"].to_numpy(dtype=float),
        )

        rows.append({
            "model_name": model_name,
            "cancer_type": cancer_type,
            "n_patients": int(len(g)),
            "n_dead": int((g["event"] == 1).sum()),
            "n_alive": int((g["event"] == 0).sum()),
            "test_cindex": compute_cindex(
                g["event"].to_numpy(dtype=float),
                g["time_days"].to_numpy(dtype=float),
                g["risk_score"].to_numpy(dtype=float),
            ),
            "test_uno_cindex": uno["uno_cindex"],
            "uno_tau_days": uno["uno_tau_days"],
            "n_test_rows_used_for_uno": uno["n_test_rows_used"],
            "median_time_days": float(g["time_days"].median()),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "cancer_type"])


def build_incremental_value_table(
    test_predictions: pd.DataFrame,
    baseline_model_name: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    baseline_df = test_predictions.loc[test_predictions["model_name"] == baseline_model_name].copy()
    if baseline_df.empty:
        return pd.DataFrame(columns=[
            "baseline_model_name",
            "model_name",
            "delta_cindex_point",
            "delta_ci_lower_95",
            "delta_ci_upper_95",
            "delta_bootstrap_mean",
            "delta_bootstrap_std",
            "n_bootstrap_valid",
            "n_paired_rows",
        ])

    rows = []
    for model_name, g in test_predictions.groupby("model_name", dropna=False):
        if model_name == baseline_model_name:
            continue

        delta = paired_bootstrap_delta_vs_baseline(
            baseline_df=baseline_df,
            model_df=g,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({
            "baseline_model_name": baseline_model_name,
            "model_name": model_name,
            **delta,
        })

    return pd.DataFrame(rows).sort_values("delta_cindex_point", ascending=False)


def merge_benchmark_tables(
    validation_results: pd.DataFrame,
    final_test_results: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    uno_table: pd.DataFrame,
) -> pd.DataFrame:
    val = validation_results.copy()
    tst = final_test_results.copy()
    boot = bootstrap_table.copy()
    uno = uno_table.copy()

    out = tst.merge(
        val[["model_name", "validation_cindex"]],
        on="model_name",
        how="left",
    ).merge(
        boot[[
            "model_name",
            "bootstrap_mean_cindex",
            "bootstrap_std_cindex",
            "ci_lower_95",
            "ci_upper_95",
            "n_bootstrap_valid",
        ]],
        on="model_name",
        how="left",
    ).merge(
        uno[[
            "model_name",
            "uno_cindex",
            "uno_tau_days",
            "n_train_ref_rows",
            "n_test_rows_used",
        ]],
        on="model_name",
        how="left",
    )

    out["validation_rank"] = out["validation_cindex"].rank(ascending=False, method="min")
    out["test_rank"] = out["test_cindex"].rank(ascending=False, method="min")
    out["generalization_gap_test_minus_val"] = out["test_cindex"] - out["validation_cindex"]

    order_cols = [
        "model_name",
        "model_family",
        "feature_set",
        "n_features",
        "validation_cindex",
        "test_cindex",
        "uno_cindex",
        "uno_tau_days",
        "n_test_rows_used",
        "generalization_gap_test_minus_val",
        "bootstrap_mean_cindex",
        "bootstrap_std_cindex",
        "ci_lower_95",
        "ci_upper_95",
        "n_bootstrap_valid",
        "validation_rank",
        "test_rank",
        "best_params_json",
        "trainval_cindex",
        "risk_sign_applied",
    ]
    existing = [c for c in order_cols if c in out.columns]
    out = out[existing].sort_values("test_cindex", ascending=False).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def make_pooled_benchmark_figure(benchmark_df: pd.DataFrame, figure_path: Path) -> None:
    plot_df = benchmark_df.sort_values("test_cindex", ascending=True).copy()

    y = np.arange(len(plot_df))
    x = plot_df["test_cindex"].to_numpy(dtype=float)
    lower = plot_df["ci_lower_95"].to_numpy(dtype=float)
    upper = plot_df["ci_upper_95"].to_numpy(dtype=float)

    xerr_left = np.where(np.isfinite(lower), x - lower, 0.0)
    xerr_right = np.where(np.isfinite(upper), upper - x, 0.0)
    xerr = np.vstack([xerr_left, xerr_right])

    plt.figure(figsize=(10, 6))
    plt.barh(y, x)
    plt.errorbar(x, y, xerr=xerr, fmt="none", capsize=3)
    plt.yticks(y, plot_df["model_name"])
    plt.xlabel("Test C-index")
    plt.ylabel("Model")
    plt.title("Pooled test benchmark with bootstrap 95% CI")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    seed = int(cfg["project"]["seed"])
    processed_dir = Path(cfg["outputs"]["processed_dir"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    predictions_dir = Path(cfg["outputs"]["predictions_dir"])
    figures_dir = Path(cfg["outputs"]["figures_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    for p in [processed_dir, tables_dir, predictions_dir, figures_dir, audit_dir]:
        ensure_dir(p)

    validation_results_path = tables_dir / "validation_results.csv"
    final_test_results_path = tables_dir / "final_test_results.csv"
    test_predictions_path = predictions_dir / "test_predictions.csv"
    bundle_path = processed_dir / "preprocessed_data_bundle.joblib"

    for p in [validation_results_path, final_test_results_path, test_predictions_path, bundle_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    validation_results = pd.read_csv(validation_results_path)
    final_test_results = pd.read_csv(final_test_results_path)
    test_predictions = pd.read_csv(test_predictions_path)
    bundle = joblib.load(bundle_path)

    validate_required_columns(
        validation_results,
        ["model_name", "validation_cindex"],
        "validation_results.csv",
    )
    validate_required_columns(
        final_test_results,
        ["model_name", "model_family", "feature_set", "n_features", "test_cindex"],
        "final_test_results.csv",
    )
    validate_required_columns(
        test_predictions,
        ["model_name", "patient_id", "cancer_type", "event", "time_days", "risk_score"],
        "test_predictions.csv",
    )

    test_predictions = normalize_prediction_table(test_predictions)
    trainval_ref_df = build_trainval_reference(bundle)

    bootstrap_n = 1000
    baseline_model_name = "Cox_ClinicalOnly"

    bootstrap_table = build_per_model_bootstrap_table(
        test_predictions=test_predictions,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )

    uno_table = build_per_model_uno_table(
        test_predictions=test_predictions,
        trainval_ref_df=trainval_ref_df,
    )

    per_cohort_table = build_per_cohort_test_table(
        test_predictions=test_predictions,
        trainval_ref_df=trainval_ref_df,
    )

    incremental_value_table = build_incremental_value_table(
        test_predictions=test_predictions,
        baseline_model_name=baseline_model_name,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )

    pooled_benchmark = merge_benchmark_tables(
        validation_results=validation_results,
        final_test_results=final_test_results,
        bootstrap_table=bootstrap_table,
        uno_table=uno_table,
    )

    pooled_benchmark_path = tables_dir / "pooled_benchmark_results.csv"
    bootstrap_path = tables_dir / "pooled_test_bootstrap_cis.csv"
    uno_path = tables_dir / "pooled_test_uno_cindices.csv"
    per_cohort_path = tables_dir / "pooled_test_by_cohort.csv"
    incremental_value_path = tables_dir / "incremental_value_vs_clinical_baseline.csv"

    pooled_benchmark.to_csv(pooled_benchmark_path, index=False)
    bootstrap_table.to_csv(bootstrap_path, index=False)
    uno_table.to_csv(uno_path, index=False)
    per_cohort_table.to_csv(per_cohort_path, index=False)
    incremental_value_table.to_csv(incremental_value_path, index=False)

    figure_path = figures_dir / "pooled_benchmark_test_cindex.png"
    make_pooled_benchmark_figure(pooled_benchmark, figure_path)

    top_model = None
    if not pooled_benchmark.empty:
        top_model = pooled_benchmark.iloc[0].to_dict()

    summary = {
        "baseline_model_name": baseline_model_name,
        "bootstrap_n": bootstrap_n,
        "secondary_metric": "uno_cindex",
        "n_models_evaluated": int(pooled_benchmark["model_name"].nunique()) if not pooled_benchmark.empty else 0,
        "top_model": top_model,
        "output_files": {
            "pooled_benchmark_results": str(pooled_benchmark_path),
            "pooled_test_bootstrap_cis": str(bootstrap_path),
            "pooled_test_uno_cindices": str(uno_path),
            "pooled_test_by_cohort": str(per_cohort_path),
            "incremental_value_vs_clinical_baseline": str(incremental_value_path),
            "pooled_benchmark_figure": str(figure_path),
        },
    }
    save_json(summary, audit_dir / "pooled_evaluation_summary.json")

    print("=" * 80)
    print("POOLED EVALUATION COMPLETE")
    print("=" * 80)
    if top_model is not None:
        print(f"Top model: {top_model['model_name']}")
        print(f"Top test C-index: {top_model['test_cindex']:.4f}")
        if "uno_cindex" in top_model and pd.notna(top_model["uno_cindex"]):
            print(f"Top Uno C-index: {top_model['uno_cindex']:.4f}")
    print(f"Pooled benchmark table: {pooled_benchmark_path}")
    print(f"Bootstrap CI table: {bootstrap_path}")
    print(f"Uno C-index table: {uno_path}")
    print(f"Per-cohort test table: {per_cohort_path}")
    print(f"Incremental value table: {incremental_value_path}")
    print(f"Figure: {figure_path}")
    print(f"Summary: {audit_dir / 'pooled_evaluation_summary.json'}")


if __name__ == "__main__":
    main()