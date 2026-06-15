from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, friedmanchisquare
from sksurv.metrics import concordance_index_censored


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


def normalize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["patient_id"] = out["patient_id"].astype(str).str.strip()
    out["cancer_type"] = out["cancer_type"].astype(str).str.strip()
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    out["time_days"] = pd.to_numeric(out["time_days"], errors="coerce")
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out = out.dropna(subset=["model_name", "patient_id", "cancer_type", "event", "time_days", "risk_score"]).copy()
    out = out.loc[out["time_days"] > 0].copy()
    return out


def normalize_loco_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["holdout_cancer"] = out["holdout_cancer"].astype(str).str.strip()
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["holdout_test_cindex"] = pd.to_numeric(out["holdout_test_cindex"], errors="coerce")
    out = out.dropna(subset=["holdout_cancer", "model_name", "holdout_test_cindex"]).copy()
    return out


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    pvals = pvals.astype(float)
    out = pd.Series(np.nan, index=pvals.index, dtype=float)

    valid = pvals.dropna()
    if valid.empty:
        return out

    m = len(valid)
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    adjusted = np.empty(m, dtype=float)

    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        adj = ranked[i] * m / rank
        prev = min(prev, adj)
        adjusted[i] = min(prev, 1.0)

    adjusted_series = pd.Series(index=valid.index[order], data=adjusted)
    out.loc[adjusted_series.index] = adjusted_series
    return out


# -----------------------------------------------------------------------------
# Paired bootstrap on pooled test predictions
# -----------------------------------------------------------------------------

def paired_bootstrap_compare(
    ref_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    merged = ref_df[["patient_id", "event", "time_days", "risk_score"]].rename(
        columns={"risk_score": "risk_ref"}
    ).merge(
        cmp_df[["patient_id", "risk_score"]].rename(columns={"risk_score": "risk_cmp"}),
        on="patient_id",
        how="inner",
    )

    if merged.empty:
        return {
            "n_paired_rows": 0,
            "ref_cindex": np.nan,
            "cmp_cindex": np.nan,
            "delta_cindex_cmp_minus_ref": np.nan,
            "delta_bootstrap_mean": np.nan,
            "delta_bootstrap_std": np.nan,
            "delta_ci_lower_95": np.nan,
            "delta_ci_upper_95": np.nan,
            "p_two_sided_bootstrap": np.nan,
            "n_bootstrap_valid": 0,
        }

    event = merged["event"].to_numpy(dtype=float)
    time_days = merged["time_days"].to_numpy(dtype=float)
    risk_ref = merged["risk_ref"].to_numpy(dtype=float)
    risk_cmp = merged["risk_cmp"].to_numpy(dtype=float)

    ref_c = compute_cindex(event, time_days, risk_ref)
    cmp_c = compute_cindex(event, time_days, risk_cmp)
    point_delta = cmp_c - ref_c

    rng = np.random.default_rng(seed)
    n = len(merged)
    deltas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            ref_b = compute_cindex(event[idx], time_days[idx], risk_ref[idx])
            cmp_b = compute_cindex(event[idx], time_days[idx], risk_cmp[idx])
            delta = cmp_b - ref_b
            if np.isfinite(delta):
                deltas.append(delta)
        except Exception:
            continue

    if len(deltas) == 0:
        return {
            "n_paired_rows": int(len(merged)),
            "ref_cindex": float(ref_c),
            "cmp_cindex": float(cmp_c),
            "delta_cindex_cmp_minus_ref": float(point_delta),
            "delta_bootstrap_mean": np.nan,
            "delta_bootstrap_std": np.nan,
            "delta_ci_lower_95": np.nan,
            "delta_ci_upper_95": np.nan,
            "p_two_sided_bootstrap": np.nan,
            "n_bootstrap_valid": 0,
        }

    deltas = np.asarray(deltas, dtype=float)
    p_left = np.mean(deltas <= 0.0)
    p_right = np.mean(deltas >= 0.0)
    p_two = min(1.0, 2.0 * min(p_left, p_right))

    return {
        "n_paired_rows": int(len(merged)),
        "ref_cindex": float(ref_c),
        "cmp_cindex": float(cmp_c),
        "delta_cindex_cmp_minus_ref": float(point_delta),
        "delta_bootstrap_mean": float(np.mean(deltas)),
        "delta_bootstrap_std": float(np.std(deltas)),
        "delta_ci_lower_95": float(np.percentile(deltas, 2.5)),
        "delta_ci_upper_95": float(np.percentile(deltas, 97.5)),
        "p_two_sided_bootstrap": float(p_two),
        "n_bootstrap_valid": int(len(deltas)),
    }


def all_pairwise_pooled_comparisons(
    test_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    model_names = sorted(test_predictions["model_name"].unique().tolist())
    rows = []

    for ref_model, cmp_model in combinations(model_names, 2):
        ref_df = test_predictions.loc[test_predictions["model_name"] == ref_model].copy()
        cmp_df = test_predictions.loc[test_predictions["model_name"] == cmp_model].copy()

        stats = paired_bootstrap_compare(
            ref_df=ref_df,
            cmp_df=cmp_df,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({
            "reference_model": ref_model,
            "comparison_model": cmp_model,
            **stats,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = benjamini_hochberg(out["p_two_sided_bootstrap"])
        out = out.sort_values("delta_cindex_cmp_minus_ref", ascending=False).reset_index(drop=True)
    return out


def pooled_vs_baseline_comparisons(
    test_predictions: pd.DataFrame,
    baseline_model_name: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    baseline_df = test_predictions.loc[test_predictions["model_name"] == baseline_model_name].copy()
    if baseline_df.empty:
        return pd.DataFrame()

    rows = []
    for cmp_model in sorted(test_predictions["model_name"].unique().tolist()):
        if cmp_model == baseline_model_name:
            continue

        cmp_df = test_predictions.loc[test_predictions["model_name"] == cmp_model].copy()
        stats = paired_bootstrap_compare(
            ref_df=baseline_df,
            cmp_df=cmp_df,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({
            "baseline_model": baseline_model_name,
            "comparison_model": cmp_model,
            **stats,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = benjamini_hochberg(out["p_two_sided_bootstrap"])
        out = out.sort_values("delta_cindex_cmp_minus_ref", ascending=False).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# LOCO summaries
# -----------------------------------------------------------------------------

def loco_rank_summary(loco_df: pd.DataFrame) -> pd.DataFrame:
    pivot = loco_df.pivot(index="holdout_cancer", columns="model_name", values="holdout_test_cindex")
    ranks = pivot.rank(axis=1, ascending=False, method="average")

    summary_rows = []
    for model_name in pivot.columns:
        vals = pivot[model_name].dropna()
        rks = ranks[model_name].dropna()
        summary_rows.append({
            "model_name": model_name,
            "n_holdouts": int(vals.shape[0]),
            "mean_holdout_cindex": float(vals.mean()),
            "std_holdout_cindex": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
            "min_holdout_cindex": float(vals.min()),
            "max_holdout_cindex": float(vals.max()),
            "mean_rank_across_holdouts": float(rks.mean()),
        })

    out = pd.DataFrame(summary_rows).sort_values(
        ["mean_holdout_cindex", "mean_rank_across_holdouts"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return out


def loco_vs_baseline_summary(
    loco_df: pd.DataFrame,
    baseline_model_name: str,
) -> pd.DataFrame:
    baseline = loco_df.loc[loco_df["model_name"] == baseline_model_name, ["holdout_cancer", "holdout_test_cindex"]].rename(
        columns={"holdout_test_cindex": "baseline_cindex"}
    )

    rows = []
    for model_name in sorted(loco_df["model_name"].unique().tolist()):
        if model_name == baseline_model_name:
            continue

        cmp_df = loco_df.loc[loco_df["model_name"] == model_name, ["holdout_cancer", "holdout_test_cindex"]].rename(
            columns={"holdout_test_cindex": "comparison_cindex"}
        )
        merged = baseline.merge(cmp_df, on="holdout_cancer", how="inner")
        if merged.empty:
            continue

        merged["delta"] = merged["comparison_cindex"] - merged["baseline_cindex"]
        wins = int((merged["delta"] > 1e-12).sum())
        losses = int((merged["delta"] < -1e-12).sum())
        ties = int((merged["delta"].abs() <= 1e-12).sum())

        n_non_ties = wins + losses
        if n_non_ties > 0:
            sign_p = float(binomtest(k=wins, n=n_non_ties, p=0.5, alternative="two-sided").pvalue)
        else:
            sign_p = np.nan

        row = {
            "baseline_model": baseline_model_name,
            "comparison_model": model_name,
            "n_holdouts_compared": int(len(merged)),
            "wins_vs_baseline": wins,
            "losses_vs_baseline": losses,
            "ties_vs_baseline": ties,
            "mean_delta_cindex": float(merged["delta"].mean()),
            "median_delta_cindex": float(merged["delta"].median()),
            "min_delta_cindex": float(merged["delta"].min()),
            "max_delta_cindex": float(merged["delta"].max()),
            "sign_test_pvalue": sign_p,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = benjamini_hochberg(out["sign_test_pvalue"])
        out = out.sort_values("mean_delta_cindex", ascending=False).reset_index(drop=True)
    return out


def friedman_test_summary(loco_df: pd.DataFrame) -> dict[str, Any]:
    pivot = loco_df.pivot(index="holdout_cancer", columns="model_name", values="holdout_test_cindex")
    pivot = pivot.dropna(axis=1, how="any")

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return {
            "friedman_statistic": np.nan,
            "friedman_pvalue": np.nan,
            "n_holdouts": int(pivot.shape[0]),
            "n_models": int(pivot.shape[1]),
            "included_models": pivot.columns.tolist(),
            "status": "insufficient_data",
        }

    arrays = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
    stat, pval = friedmanchisquare(*arrays)

    return {
        "friedman_statistic": float(stat),
        "friedman_pvalue": float(pval),
        "n_holdouts": int(pivot.shape[0]),
        "n_models": int(pivot.shape[1]),
        "included_models": pivot.columns.tolist(),
        "status": "ok",
    }


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def make_vs_baseline_delta_figure(df: pd.DataFrame, figure_path: Path) -> None:
    if df.empty:
        return

    plot_df = df.sort_values("delta_cindex_cmp_minus_ref", ascending=True).copy()
    y = np.arange(len(plot_df))
    x = plot_df["delta_cindex_cmp_minus_ref"].to_numpy(dtype=float)
    lower = plot_df["delta_ci_lower_95"].to_numpy(dtype=float)
    upper = plot_df["delta_ci_upper_95"].to_numpy(dtype=float)

    xerr_left = np.where(np.isfinite(lower), x - lower, 0.0)
    xerr_right = np.where(np.isfinite(upper), upper - x, 0.0)
    xerr = np.vstack([xerr_left, xerr_right])

    plt.figure(figsize=(10, 6))
    plt.barh(y, x)
    plt.axvline(0.0, linestyle="--")
    plt.errorbar(x, y, xerr=xerr, fmt="none", capsize=3)
    plt.yticks(y, plot_df["comparison_model"])
    plt.xlabel("Delta C-index vs Cox_ClinicalOnly")
    plt.ylabel("Model")
    plt.title("Pooled paired bootstrap delta vs clinical baseline")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    seed = int(cfg["project"]["seed"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    predictions_dir = Path(cfg["outputs"]["predictions_dir"])
    figures_dir = Path(cfg["outputs"]["figures_dir"])
    audit_dir = Path(cfg["outputs"]["audit_dir"])

    for p in [tables_dir, predictions_dir, figures_dir, audit_dir]:
        ensure_dir(p)

    test_predictions_path = predictions_dir / "test_predictions.csv"
    loco_results_path = tables_dir / "leave_one_cancer_out_results.csv"
    pooled_benchmark_path = tables_dir / "pooled_benchmark_results.csv"

    for p in [test_predictions_path, loco_results_path, pooled_benchmark_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    test_predictions = normalize_prediction_df(pd.read_csv(test_predictions_path))
    loco_df = normalize_loco_df(pd.read_csv(loco_results_path))
    pooled_benchmark = pd.read_csv(pooled_benchmark_path)

    bootstrap_n = 2000
    baseline_model_name = "Cox_ClinicalOnly"

    all_pairwise = all_pairwise_pooled_comparisons(
        test_predictions=test_predictions,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )
    vs_baseline = pooled_vs_baseline_comparisons(
        test_predictions=test_predictions,
        baseline_model_name=baseline_model_name,
        n_bootstrap=bootstrap_n,
        seed=seed,
    )
    loco_rank = loco_rank_summary(loco_df)
    loco_vs_baseline = loco_vs_baseline_summary(
        loco_df=loco_df,
        baseline_model_name=baseline_model_name,
    )
    friedman_summary = friedman_test_summary(loco_df)

    out_all_pairwise = tables_dir / "statistical_pairwise_pooled_bootstrap.csv"
    out_vs_baseline = tables_dir / "statistical_vs_clinical_baseline_pooled.csv"
    out_loco_rank = tables_dir / "statistical_leave_one_cancer_out_rank_summary.csv"
    out_loco_vs_base = tables_dir / "statistical_leave_one_cancer_out_vs_clinical.csv"
    out_friedman = audit_dir / "statistical_friedman_summary.json"
    out_fig = figures_dir / "statistical_delta_vs_clinical_baseline.png"

    all_pairwise.to_csv(out_all_pairwise, index=False)
    vs_baseline.to_csv(out_vs_baseline, index=False)
    loco_rank.to_csv(out_loco_rank, index=False)
    loco_vs_baseline.to_csv(out_loco_vs_base, index=False)
    save_json(friedman_summary, out_friedman)
    make_vs_baseline_delta_figure(vs_baseline, out_fig)

    top_model = None
    if not pooled_benchmark.empty:
        top_model = pooled_benchmark.sort_values("test_cindex", ascending=False).iloc[0]["model_name"]

    summary = {
        "bootstrap_n": bootstrap_n,
        "baseline_model_name": baseline_model_name,
        "top_pooled_model": top_model,
        "n_models_pooled": int(test_predictions["model_name"].nunique()),
        "n_models_loco": int(loco_df["model_name"].nunique()),
        "friedman_summary": friedman_summary,
        "outputs": {
            "pairwise_pooled_bootstrap": str(out_all_pairwise),
            "vs_clinical_baseline_pooled": str(out_vs_baseline),
            "leave_one_cancer_out_rank_summary": str(out_loco_rank),
            "leave_one_cancer_out_vs_clinical": str(out_loco_vs_base),
            "friedman_summary": str(out_friedman),
            "delta_vs_clinical_figure": str(out_fig),
        },
    }
    save_json(summary, audit_dir / "statistical_comparison_summary.json")

    print("=" * 80)
    print("STATISTICAL COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Pairwise pooled bootstrap table: {out_all_pairwise}")
    print(f"Baseline comparison table: {out_vs_baseline}")
    print(f"LOCO rank summary: {out_loco_rank}")
    print(f"LOCO vs baseline summary: {out_loco_vs_base}")
    print(f"Friedman summary: {out_friedman}")
    print(f"Figure: {out_fig}")
    print(f"Summary: {audit_dir / 'statistical_comparison_summary.json'}")


if __name__ == "__main__":
    main()