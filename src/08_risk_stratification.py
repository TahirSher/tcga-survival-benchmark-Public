from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
from lifelines.exceptions import ConvergenceWarning

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


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# -----------------------------------------------------------------------------
# Data loading / normalization
# -----------------------------------------------------------------------------

def normalize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["cancer_type"] = out["cancer_type"].astype(str).str.strip()
    out["event"] = to_numeric_series(out["event"])
    out["time_days"] = to_numeric_series(out["time_days"])
    out["risk_score"] = to_numeric_series(out["risk_score"])
    out = out.dropna(subset=["model_name", "split", "patient_id", "cancer_type", "event", "time_days", "risk_score"]).copy()
    out = out.loc[out["time_days"] > 0].copy()
    return out


def choose_models(
    pooled_benchmark: pd.DataFrame,
    top_k: int = 3,
) -> list[str]:
    if pooled_benchmark.empty:
        return []
    return pooled_benchmark.sort_values("test_cindex", ascending=False)["model_name"].head(top_k).tolist()


# -----------------------------------------------------------------------------
# Risk grouping
# -----------------------------------------------------------------------------

def training_quantile_thresholds(trainval_df: pd.DataFrame) -> tuple[float, float]:
    q1 = float(trainval_df["risk_score"].quantile(1.0 / 3.0))
    q2 = float(trainval_df["risk_score"].quantile(2.0 / 3.0))
    return q1, q2


def assign_three_risk_groups(scores: pd.Series, q1: float, q2: float) -> pd.Series:
    groups = pd.Series(index=scores.index, dtype="object")
    groups.loc[scores <= q1] = "Low"
    groups.loc[(scores > q1) & (scores <= q2)] = "Intermediate"
    groups.loc[scores > q2] = "High"
    return groups


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def overall_logrank_three_group(df: pd.DataFrame) -> dict[str, float]:
    result = multivariate_logrank_test(
        event_durations=df["time_days"],
        groups=df["risk_group"],
        event_observed=df["event"],
    )
    return {
        "overall_logrank_statistic": float(result.test_statistic),
        "overall_logrank_pvalue": float(result.p_value),
    }


def pairwise_logrank(df: pd.DataFrame, g1: str, g2: str) -> dict[str, float]:
    temp = df.loc[df["risk_group"].isin([g1, g2])].copy()
    a = temp.loc[temp["risk_group"] == g1]
    b = temp.loc[temp["risk_group"] == g2]

    if len(a) == 0 or len(b) == 0:
        return {
            "group_1": g1,
            "group_2": g2,
            "n_group_1": int(len(a)),
            "n_group_2": int(len(b)),
            "logrank_statistic": np.nan,
            "logrank_pvalue": np.nan,
        }

    result = logrank_test(
        durations_A=a["time_days"],
        durations_B=b["time_days"],
        event_observed_A=a["event"],
        event_observed_B=b["event"],
    )
    return {
        "group_1": g1,
        "group_2": g2,
        "n_group_1": int(len(a)),
        "n_group_2": int(len(b)),
        "logrank_statistic": float(result.test_statistic),
        "logrank_pvalue": float(result.p_value),
    }


def pairwise_hazard_ratio(df: pd.DataFrame, g_ref: str, g_cmp: str) -> dict[str, float | str]:
    temp = df.loc[df["risk_group"].isin([g_ref, g_cmp])].copy()

    if temp.empty:
        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "n_rows": 0,
            "n_ref": 0,
            "n_cmp": 0,
            "n_events_ref": 0,
            "n_events_cmp": 0,
            "hr_status": "empty_comparison",
        }

    temp["indicator"] = (temp["risk_group"] == g_cmp).astype(int)

    ref_df = temp.loc[temp["risk_group"] == g_ref].copy()
    cmp_df = temp.loc[temp["risk_group"] == g_cmp].copy()

    n_ref = int(len(ref_df))
    n_cmp = int(len(cmp_df))
    n_events_ref = int((ref_df["event"] == 1).sum())
    n_events_cmp = int((cmp_df["event"] == 1).sum())
    n_nonevents_ref = int((ref_df["event"] == 0).sum())
    n_nonevents_cmp = int((cmp_df["event"] == 0).sum())

    # Minimum stability rules
    if n_ref < 10 or n_cmp < 10:
        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "n_rows": int(len(temp)),
            "n_ref": n_ref,
            "n_cmp": n_cmp,
            "n_events_ref": n_events_ref,
            "n_events_cmp": n_events_cmp,
            "hr_status": "skipped_small_group",
        }

    if n_events_ref < 3 or n_events_cmp < 3:
        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "n_rows": int(len(temp)),
            "n_ref": n_ref,
            "n_cmp": n_cmp,
            "n_events_ref": n_events_ref,
            "n_events_cmp": n_events_cmp,
            "hr_status": "skipped_too_few_events",
        }

    if n_nonevents_ref < 3 or n_nonevents_cmp < 3:
        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "n_rows": int(len(temp)),
            "n_ref": n_ref,
            "n_cmp": n_cmp,
            "n_events_ref": n_events_ref,
            "n_events_cmp": n_events_cmp,
            "hr_status": "skipped_too_few_nonevents",
        }

    try:
        # Penalized Cox for stability
        cph = CoxPHFitter(penalizer=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            cph.fit(
                temp[["time_days", "event", "indicator"]],
                duration_col="time_days",
                event_col="event"
            )

        coef = float(cph.params_["indicator"])
        ci_low_log = float(cph.confidence_intervals_.loc["indicator", "95% lower-bound"])
        ci_up_log = float(cph.confidence_intervals_.loc["indicator", "95% upper-bound"])
        pval = float(cph.summary.loc["indicator", "p"])

        # Clip exponentiation to avoid overflow in pathological edge cases
        hr = float(np.exp(np.clip(coef, -20, 20)))
        ci_low = float(np.exp(np.clip(ci_low_log, -20, 20)))
        ci_up = float(np.exp(np.clip(ci_up_log, -20, 20)))

        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": hr,
            "ci_lower_95": ci_low,
            "ci_upper_95": ci_up,
            "p_value": pval,
            "n_rows": int(len(temp)),
            "n_ref": n_ref,
            "n_cmp": n_cmp,
            "n_events_ref": n_events_ref,
            "n_events_cmp": n_events_cmp,
            "hr_status": "ok_penalized",
        }

    except Exception:
        return {
            "reference_group": g_ref,
            "comparison_group": g_cmp,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "n_rows": int(len(temp)),
            "n_ref": n_ref,
            "n_cmp": n_cmp,
            "n_events_ref": n_events_ref,
            "n_events_cmp": n_events_cmp,
            "hr_status": "fit_failed",
        }


def summarize_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["risk_group"], dropna=False)
        .agg(
            n_patients=("patient_id", "count"),
            n_dead=("event", lambda s: int((s == 1).sum())),
            n_alive=("event", lambda s: int((s == 0).sum())),
            median_time_days=("time_days", "median"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )
    return out


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def km_curve_three_groups(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))

    order = ["Low", "Intermediate", "High"]
    kmf = KaplanMeierFitter()

    for grp in order:
        g = df.loc[df["risk_group"] == grp]
        if g.empty:
            continue
        kmf.fit(durations=g["time_days"], event_observed=g["event"], label=grp)
        kmf.plot(ci_show=False)

    plt.title(title)
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    tables_dir = Path(cfg["outputs"]["tables_dir"])
    predictions_dir = Path(cfg["outputs"]["predictions_dir"])
    figures_dir = Path(cfg["outputs"]["figures_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    for p in [tables_dir, predictions_dir, figures_dir, audit_dir]:
        ensure_dir(p)

    pooled_benchmark_path = tables_dir / "pooled_benchmark_results.csv"
    trainval_predictions_path = predictions_dir / "trainval_predictions.csv"
    test_predictions_path = predictions_dir / "test_predictions.csv"

    for p in [pooled_benchmark_path, trainval_predictions_path, test_predictions_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    pooled_benchmark = pd.read_csv(pooled_benchmark_path)
    trainval_predictions = normalize_prediction_df(pd.read_csv(trainval_predictions_path))
    test_predictions = normalize_prediction_df(pd.read_csv(test_predictions_path))

    top_models = choose_models(pooled_benchmark, top_k=3)
    if not top_models:
        raise ValueError("No models found in pooled benchmark results.")

    all_pairwise_rows = []
    all_hr_rows = []
    all_group_count_rows = []
    all_threshold_rows = []
    all_overall_rows = []
    all_test_assignments = []

    for model_name in top_models:
        trv = trainval_predictions.loc[trainval_predictions["model_name"] == model_name].copy()
        tst = test_predictions.loc[test_predictions["model_name"] == model_name].copy()

        if trv.empty or tst.empty:
            continue

        q1, q2 = training_quantile_thresholds(trv)
        tst["risk_group"] = assign_three_risk_groups(tst["risk_score"], q1=q1, q2=q2)

        all_test_assignments.append(tst.copy())

        all_threshold_rows.append({
            "model_name": model_name,
            "trainval_q1_threshold": q1,
            "trainval_q2_threshold": q2,
            "n_trainval": int(len(trv)),
            "n_test": int(len(tst)),
        })

        # Overall pooled test
        overall_stats = overall_logrank_three_group(tst)
        all_overall_rows.append({
            "model_name": model_name,
            "scope": "ALL",
            "cancer_type": "ALL",
            **overall_stats,
        })

        pair_specs = [("Low", "Intermediate"), ("Intermediate", "High"), ("Low", "High")]

        for g1, g2 in pair_specs:
            pr = pairwise_logrank(tst, g1, g2)
            pr.update({
                "model_name": model_name,
                "scope": "ALL",
                "cancer_type": "ALL",
            })
            all_pairwise_rows.append(pr)

            hr = pairwise_hazard_ratio(tst, g1, g2)
            hr.update({
                "model_name": model_name,
                "scope": "ALL",
                "cancer_type": "ALL",
            })
            all_hr_rows.append(hr)

        gc = summarize_group_counts(tst)
        gc["model_name"] = model_name
        gc["scope"] = "ALL"
        gc["cancer_type"] = "ALL"
        all_group_count_rows.append(gc)

        km_curve_three_groups(
            df=tst,
            title=f"{model_name} - pooled test risk stratification",
            output_path=figures_dir / f"km_{model_name}_pooled_test.png",
        )

        # Cohort-wise test
        for cancer_type, g in tst.groupby("cancer_type", dropna=False):
            if g["risk_group"].nunique() < 2:
                continue

            overall_stats = overall_logrank_three_group(g)
            all_overall_rows.append({
                "model_name": model_name,
                "scope": "COHORT",
                "cancer_type": cancer_type,
                **overall_stats,
            })

            for g1, g2 in pair_specs:
                pr = pairwise_logrank(g, g1, g2)
                pr.update({
                    "model_name": model_name,
                    "scope": "COHORT",
                    "cancer_type": cancer_type,
                })
                all_pairwise_rows.append(pr)

                hr = pairwise_hazard_ratio(g, g1, g2)
                hr.update({
                    "model_name": model_name,
                    "scope": "COHORT",
                    "cancer_type": cancer_type,
                })
                all_hr_rows.append(hr)

            gc = summarize_group_counts(g)
            gc["model_name"] = model_name
            gc["scope"] = "COHORT"
            gc["cancer_type"] = cancer_type
            all_group_count_rows.append(gc)

            km_curve_three_groups(
                df=g,
                title=f"{model_name} - {cancer_type} test risk stratification",
                output_path=figures_dir / f"km_{model_name}_{cancer_type}_test.png",
            )

    if not all_test_assignments:
        raise ValueError("No risk assignments were generated.")

    thresholds_df = pd.DataFrame(all_threshold_rows)
    overall_df = pd.DataFrame(all_overall_rows)
    pairwise_df = pd.DataFrame(all_pairwise_rows)
    hr_df = pd.DataFrame(all_hr_rows)
    group_counts_df = pd.concat(all_group_count_rows, axis=0, ignore_index=True)
    test_assignments_df = pd.concat(all_test_assignments, axis=0, ignore_index=True)

    thresholds_df.to_csv(tables_dir / "risk_group_thresholds.csv", index=False)
    overall_df.to_csv(tables_dir / "risk_stratification_overall_logrank.csv", index=False)
    pairwise_df.to_csv(tables_dir / "risk_stratification_pairwise_logrank.csv", index=False)
    hr_df.to_csv(tables_dir / "risk_stratification_pairwise_hazard_ratios.csv", index=False)
    group_counts_df.to_csv(tables_dir / "risk_stratification_group_counts.csv", index=False)
    test_assignments_df.to_csv(predictions_dir / "test_predictions_with_risk_groups.csv", index=False)

    summary = {
        "top_models_used": top_models,
        "output_files": {
            "thresholds": str(tables_dir / "risk_group_thresholds.csv"),
            "overall_logrank": str(tables_dir / "risk_stratification_overall_logrank.csv"),
            "pairwise_logrank": str(tables_dir / "risk_stratification_pairwise_logrank.csv"),
            "pairwise_hazard_ratios": str(tables_dir / "risk_stratification_pairwise_hazard_ratios.csv"),
            "group_counts": str(tables_dir / "risk_stratification_group_counts.csv"),
            "test_predictions_with_risk_groups": str(predictions_dir / "test_predictions_with_risk_groups.csv"),
        },
    }
    save_json(summary, audit_dir / "risk_stratification_summary.json")

    print("=" * 80)
    print("RISK STRATIFICATION COMPLETE")
    print("=" * 80)
    print(f"Top models used: {', '.join(top_models)}")
    print(f"Threshold table: {tables_dir / 'risk_group_thresholds.csv'}")
    print(f"Overall log-rank table: {tables_dir / 'risk_stratification_overall_logrank.csv'}")
    print(f"Pairwise log-rank table: {tables_dir / 'risk_stratification_pairwise_logrank.csv'}")
    print(f"Pairwise hazard ratio table: {tables_dir / 'risk_stratification_pairwise_hazard_ratios.csv'}")
    print(f"Group counts table: {tables_dir / 'risk_stratification_group_counts.csv'}")
    print(f"Predictions with risk groups: {predictions_dir / 'test_predictions_with_risk_groups.csv'}")
    print(f"Summary: {audit_dir / 'risk_stratification_summary.json'}")


if __name__ == "__main__":
    main()