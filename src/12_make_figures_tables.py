from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


CONFIG_PATH = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project\configs\config.yaml")


# =============================================================================
# Global style
# =============================================================================

# Priority palette: Bright & Clear Blue, then Blue Ocean Breeze
BRIGHT_BLUE = ["#22AFFB", "#50BEFD", "#8CD6FF", "#B2E4FF"]
OCEAN_BREEZE = ["#34C6C1", "#4ADDCE", "#ABE9DB", "#D2F4E5", "#AAE4E8"]
LILAC_PINK = ["#CDACF2", "#DCC4F5", "#F2E4E7", "#F2D3DC", "#F2C4D5"]
RED_YELLOW_SUNSET = ["#A52222", "#D54536", "#DF6339", "#E29E39", "#F1C461"]

COLOR_MODEL_COMPACTNESS = {
    "GradientBoostingSurvival_FullAvailable": RED_YELLOW_SUNSET[0],
    "Cox_FullAvailable": RED_YELLOW_SUNSET[1],
    "ElasticNetCox_FullAvailable": RED_YELLOW_SUNSET[2],
    "RandomSurvivalForest_FullAvailable": RED_YELLOW_SUNSET[3],
    "SurvivalSVM_FullAvailable": RED_YELLOW_SUNSET[4],
}

COLOR_MODEL = {
    "Cox_ClinicalOnly": BRIGHT_BLUE[1],
    "Cox_FullAvailable": OCEAN_BREEZE[0],
    "ElasticNetCox_FullAvailable": BRIGHT_BLUE[0],
    "RandomSurvivalForest_FullAvailable": OCEAN_BREEZE[1],
    "GradientBoostingSurvival_FullAvailable": BRIGHT_BLUE[0],
    "SurvivalSVM_FullAvailable": OCEAN_BREEZE[2],
}

RISK_COLORS = {
    "Low": RED_YELLOW_SUNSET[4],
    "Intermediate": RED_YELLOW_SUNSET[2],
    "High": RED_YELLOW_SUNSET[0],
}

FIG_W = 13.333   # 16:9 aspect ratio
FIG_H = 7.5
FIG_DPI = 400

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.0,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "savefig.dpi": FIG_DPI,
    "figure.dpi": FIG_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def apply_axis_style(ax, use_grid: bool = True, tick_direction: str = "inout") -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        which="major",
        direction=tick_direction,
        length=6,
        width=1.0,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction=tick_direction,
        length=3,
        width=0.8,
        top=False,
        right=False,
    )

    if use_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


def maybe_float(x: Any) -> float | None:
    try:
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def fmt_num(x: Any, digits: int = 4, na: str = "--") -> str:
    val = maybe_float(x)
    if val is None:
        return na
    return f"{val:.{digits}f}"


def fmt_pct_from_fraction(x: Any, digits: int = 1, na: str = "--") -> str:
    val = maybe_float(x)
    if val is None:
        return na
    return f"{100.0 * val:.{digits}f}"


def fmt_p(x: Any, na: str = "--") -> str:
    val = maybe_float(x)
    if val is None:
        return na
    if val < 1e-4:
        return "<0.0001"
    return f"{val:.4f}"


def humanize_model_name(name: str) -> str:
    mapping = {
        "Cox_ClinicalOnly": "Cox (Clinical Only)",
        "Cox_FullAvailable": "Cox (Full Available)",
        "ElasticNetCox_FullAvailable": "Elastic Net Cox",
        "RandomSurvivalForest_FullAvailable": "Random Survival Forest",
        "GradientBoostingSurvival_FullAvailable": "Gradient Boosting Survival",
        "SurvivalSVM_FullAvailable": "Survival SVM",
    }
    return mapping.get(name, name.replace("_", " "))


def humanize_feature_set(name: str) -> str:
    mapping = {
        "clinical_only": "Clinical Only",
        "full_available": "Full Available",
        "biospecimen_only": "Biospecimen Only",
        "genomic_only": "Genomic Only",
    }
    return mapping.get(name, name.replace("_", " ").title())


def latex_escape(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def dataframe_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    table_env: str = "table*",
    column_format: str | None = None,
    note: str | None = None,
) -> str:
    if column_format is None:
        column_format = "l" + "c" * (df.shape[1] - 1)

    lines = [
        rf"\begin{{{table_env}}}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        vals = [latex_escape(v) for v in row.tolist()]
        lines.append(" & ".join(vals) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    if note:
        lines.extend([
            r"\vspace{1mm}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize",
            r"\textit{Note:} " + latex_escape(note),
            r"\end{minipage}",
        ])

    lines.append(rf"\end{{{table_env}}}")
    return "\n".join(lines)


def save_table_bundle(
    df: pd.DataFrame,
    out_csv: Path,
    out_tex: Path,
    caption: str,
    label: str,
    note: str | None = None,
    table_env: str | None = None,
    column_format: str | None = None,
) -> None:
    ensure_dir(out_csv.parent)
    ensure_dir(out_tex.parent)
    df.to_csv(out_csv, index=False)

    if table_env is None:
        table_env = "table*" if df.shape[1] >= 6 else "table"

    latex_str = dataframe_to_latex(
        df=df,
        caption=caption,
        label=label,
        table_env=table_env,
        column_format=column_format,
        note=note,
    )

    with out_tex.open("w", encoding="utf-8") as f:
        f.write(latex_str)


# =============================================================================
# Figure generation
# =============================================================================

def draw_workflow_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.68, 0.17, 0.18, "1. Raw Cohorts", "BRCA, COAD, LUAD\nclinical, biospecimen,\nand available omics", BRIGHT_BLUE[3]),
        (0.24, 0.68, 0.17, 0.18, "2. Master Table", "patient-level merge,\nclean survival target,\nand feature harmonization", BRIGHT_BLUE[2]),
        (0.45, 0.68, 0.17, 0.18, "3. Data Audit", "eligibility,\nmissingness,\nand cohort summary", OCEAN_BREEZE[4]),
        (0.66, 0.68, 0.17, 0.18, "4. Frozen Protocol", "train/validation/test split\nplus leave-one-cancer-out\nevaluation design", OCEAN_BREEZE[3]),
        (0.79, 0.39, 0.17, 0.18, "5. Preprocessing", "train-only imputation,\nencoding,\nfeature construction", BRIGHT_BLUE[2]),
        (0.58, 0.39, 0.17, 0.18, "6. Pooled Benchmark", "clinical baseline and\nfull-available survival models", BRIGHT_BLUE[1]),
        (0.37, 0.39, 0.17, 0.18, "7. LOCO Evaluation", "cross-cancer transfer\nstress test", OCEAN_BREEZE[2]),
        (0.16, 0.39, 0.17, 0.18, "8. Compactness", "top-k feature saturation\nand parsimonious modeling", OCEAN_BREEZE[1]),
        (0.03, 0.10, 0.30, 0.18, "9. Translational Analysis", "risk stratification, log-rank testing,\npaired bootstrap comparison,\nand manuscript-ready packaging", BRIGHT_BLUE[0]),
    ]

    for x, y, w, h, title, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#2A2A2A",
            facecolor=color,
            alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, body, ha="center", va="center",
                fontsize=10.5, fontweight="bold", linespacing=1.2)

    arrow_specs = [
        ((0.20, 0.77), (0.24, 0.77)),
        ((0.41, 0.77), (0.45, 0.77)),
        ((0.62, 0.77), (0.66, 0.77)),
        ((0.83, 0.68), (0.865, 0.57)),
        ((0.79, 0.48), (0.75, 0.48)),
        ((0.58, 0.48), (0.54, 0.48)),
        ((0.37, 0.48), (0.33, 0.48)),
        ((0.16, 0.39), (0.16, 0.28)),
    ]

    for start, end in arrow_specs:
        arrow = FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=16,
            linewidth=1.1, color="#2A2A2A"
        )
        ax.add_patch(arrow)

    ax.set_title("Leakage-Safe Cross-Cohort Survival Benchmarking Workflow",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_pooled_benchmark_figure(pooled_benchmark: pd.DataFrame, out_path: Path) -> None:
    df = pooled_benchmark.copy()
    df["Model"] = df["model_name"].map(humanize_model_name)
    df = df.sort_values("test_cindex", ascending=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    y = np.arange(len(df))
    x = df["test_cindex"].to_numpy(dtype=float)
    lo = df["ci_lower_95"].to_numpy(dtype=float)
    hi = df["ci_upper_95"].to_numpy(dtype=float)

    xerr = np.vstack([x - lo, hi - x])

    colors = [COLOR_MODEL.get(m, BRIGHT_BLUE[1]) for m in df["model_name"]]
    ax.barh(y, x, color=colors, edgecolor="#2A2A2A", linewidth=0.8)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#1A1A1A", elinewidth=1.0, capsize=4)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.set_xlabel("Test C-index", fontweight="bold")
    ax.set_ylabel("Model", fontweight="bold")
    ax.set_title("Pooled Test Benchmark with Bootstrap 95% Confidence Intervals", fontweight="bold")

    apply_axis_style(ax, use_grid=True, tick_direction="inout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_loco_heatmap(loco_results: pd.DataFrame, out_path: Path) -> None:
    pivot = loco_results.pivot(index="model_name", columns="holdout_cancer", values="holdout_test_cindex")
    model_order = [
        "Cox_ClinicalOnly",
        "Cox_FullAvailable",
        "ElasticNetCox_FullAvailable",
        "GradientBoostingSurvival_FullAvailable",
        "RandomSurvivalForest_FullAvailable",
        "SurvivalSVM_FullAvailable",
    ]
    ordered_index = [m for m in model_order if m in pivot.index] + [m for m in pivot.index if m not in model_order]
    pivot = pivot.loc[ordered_index]

    mat = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=np.nanmin(mat), vmax=np.nanmax(mat))

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), fontweight="bold")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([humanize_model_name(x) for x in pivot.index.tolist()], fontweight="bold")
    ax.set_xlabel("Held-Out Cancer Cohort", fontweight="bold")
    ax.set_ylabel("Model", fontweight="bold")
    ax.set_title("Leave-One-Cancer-Out Test C-index Heatmap", fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color="black", fontsize=10.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.set_ylabel("Test C-index", rotation=270, labelpad=16, fontweight="bold")
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")

    apply_axis_style(ax, use_grid=False, tick_direction="out")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_compactness_figure(compactness: pd.DataFrame, out_path: Path) -> None:
    model_order = [
        "GradientBoostingSurvival_FullAvailable",
        "Cox_FullAvailable",
        "ElasticNetCox_FullAvailable",
        "RandomSurvivalForest_FullAvailable",
        "SurvivalSVM_FullAvailable",
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for model_name in model_order:
        if model_name not in compactness["model_name"].unique():
            continue
        g = compactness.loc[compactness["model_name"] == model_name].copy().sort_values("n_features")
        ax.plot(
            g["n_features"],
            g["test_cindex"],
            marker="o",
            markersize=5.5,
            linewidth=2.2,
            color=COLOR_MODEL_COMPACTNESS.get(model_name, RED_YELLOW_SUNSET[2]),
            label=humanize_model_name(model_name),
        )

    ax.set_xlabel("Number of Top-Ranked Features", fontweight="bold")
    ax.set_ylabel("Test C-index", fontweight="bold")
    ax.set_title("Feature Compactness and Performance Saturation", fontweight="bold")
    ax.set_xticks(sorted(compactness["n_features"].unique().tolist()))

    apply_axis_style(ax, use_grid=True, tick_direction="inout")
    leg = ax.legend(title="Model", loc="lower right")
    if leg is not None:
        leg.get_title().set_fontweight("bold")
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_km_figure(test_predictions_risk: pd.DataFrame, best_model: str, out_path: Path) -> None:
    df = test_predictions_risk.loc[test_predictions_risk["model_name"] == best_model].copy()
    if df.empty:
        raise ValueError(f"No risk-stratified predictions found for best model: {best_model}")

    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df = df.dropna(subset=["event", "time_days", "risk_group"]).copy()
    df = df.loc[df["time_days"] > 0].copy()

    groups = ["Low", "Intermediate", "High"]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    kmf = KaplanMeierFitter()

    for grp in groups:
        g = df.loc[df["risk_group"] == grp].copy()
        if g.empty:
            continue
        kmf.fit(g["time_days"], event_observed=g["event"], label=grp)
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=RISK_COLORS.get(grp, BRIGHT_BLUE[1]),
            linewidth=2.4,
        )

    logrank_result = multivariate_logrank_test(
        event_durations=df["time_days"],
        groups=df["risk_group"],
        event_observed=df["event"],
    )

    pval = logrank_result.p_value
    p_txt = "<0.0001" if pval < 1e-4 else f"{pval:.4f}"

    ax.set_xlabel("Time (days)", fontweight="bold")
    ax.set_ylabel("Survival Probability", fontweight="bold")
    ax.set_title(
        f"Pooled Test Kaplan–Meier Curves by {humanize_model_name(best_model)} Risk Group",
        fontweight="bold"
    )

    ax.text(
        0.98, 0.95,
        f"Overall log-rank p = {p_txt}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2A2A2A", alpha=0.95),
    )

    apply_axis_style(ax, use_grid=False, tick_direction="inout")
    leg = ax.legend(title="Risk Group", loc="lower left")
    if leg is not None:
        leg.get_title().set_fontweight("bold")
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def draw_delta_vs_baseline_figure(stat_vs_baseline: pd.DataFrame, out_path: Path) -> None:
    df = stat_vs_baseline.copy()
    df["Model"] = df["comparison_model"].map(humanize_model_name)
    df = df.sort_values("delta_cindex_cmp_minus_ref", ascending=True)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    y = np.arange(len(df))
    x = df["delta_cindex_cmp_minus_ref"].to_numpy(dtype=float)
    lo = df["delta_ci_lower_95"].to_numpy(dtype=float)
    hi = df["delta_ci_upper_95"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])

    colors = [COLOR_MODEL.get(m, BRIGHT_BLUE[1]) for m in df["comparison_model"]]
    ax.barh(y, x, color=colors, edgecolor="#2A2A2A", linewidth=0.8)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#1A1A1A", elinewidth=1.0, capsize=4)
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="#1A1A1A")

    ax.set_yticks(y)
    ax.set_yticklabels(df["Model"])
    ax.set_xlabel(r"$\Delta$ C-index versus Cox (Clinical Only)", fontweight="bold")
    ax.set_ylabel("Comparison Model", fontweight="bold")
    ax.set_title("Pooled Paired Bootstrap Comparison versus Clinical Baseline", fontweight="bold")

    apply_axis_style(ax, use_grid=True, tick_direction="inout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Table preparation
# =============================================================================

def prepare_table_1(audit_summary: pd.DataFrame) -> pd.DataFrame:
    df = audit_summary.copy()

    keep_cols = [
        "cancer_type",
        "n_unique_patients",
        "n_dead",
        "n_alive",
        "dead_rate",
        "n_usable_for_survival_modeling",
        "median_time_years_valid",
        "mean_age_years",
        "omics_present_any",
    ]
    df = df[keep_cols].copy()

    out = pd.DataFrame({
        "Cohort": df["cancer_type"],
        "Patients": df["n_unique_patients"].astype(int).astype(str),
        "Dead": df["n_dead"].astype(int).astype(str),
        "Alive": df["n_alive"].astype(int).astype(str),
        "Dead Rate (\\%)": df["dead_rate"].apply(lambda x: fmt_pct_from_fraction(x, 1)),
        "Usable for Survival Modeling": df["n_usable_for_survival_modeling"].astype(int).astype(str),
        "Median Follow-Up (Years)": df["median_time_years_valid"].apply(lambda x: fmt_num(x, 2)),
        "Mean Age (Years)": df["mean_age_years"].apply(lambda x: fmt_num(x, 2)),
        "Omics Available": df["omics_present_any"].map({True: "Yes", False: "No"}).fillna("No"),
    })
    return out


def prepare_table_2(pooled_benchmark: pd.DataFrame) -> pd.DataFrame:
    df = pooled_benchmark.copy()
    df = df.sort_values("test_cindex", ascending=False)

    out = pd.DataFrame({
        "Model": df["model_name"].map(humanize_model_name),
        "Feature Set": df["feature_set"].map(humanize_feature_set),
        "Features": df["n_features"].astype(int).astype(str),
        "Validation C-index": df["validation_cindex"].apply(lambda x: fmt_num(x, 4)),
        "Test C-index": df["test_cindex"].apply(lambda x: fmt_num(x, 4)),
        "95\\% CI": [
            f"{fmt_num(lo, 4)} to {fmt_num(hi, 4)}"
            for lo, hi in zip(df["ci_lower_95"], df["ci_upper_95"])
        ],
        "Test minus Validation": df["generalization_gap_test_minus_val"].apply(lambda x: fmt_num(x, 4)),
    })
    return out


def prepare_table_3_loco(loco_results: pd.DataFrame) -> pd.DataFrame:
    pivot = loco_results.pivot(index="model_name", columns="holdout_cancer", values="holdout_test_cindex")
    pivot["Mean"] = pivot.mean(axis=1)

    model_order = [
        "Cox_ClinicalOnly",
        "Cox_FullAvailable",
        "ElasticNetCox_FullAvailable",
        "GradientBoostingSurvival_FullAvailable",
        "RandomSurvivalForest_FullAvailable",
        "SurvivalSVM_FullAvailable",
    ]
    ordered_index = [m for m in model_order if m in pivot.index] + [m for m in pivot.index if m not in model_order]
    pivot = pivot.loc[ordered_index]

    cols = [c for c in ["BRCA", "COAD", "LUAD", "Mean"] if c in pivot.columns]

    out = pd.DataFrame({"Model": [humanize_model_name(m) for m in pivot.index]})
    for c in cols:
        out[c] = pivot[c].apply(lambda x: fmt_num(x, 4))
    return out


def prepare_table_4_compactness(plateau_summary: pd.DataFrame) -> pd.DataFrame:
    df = plateau_summary.copy()
    df = df.sort_values("best_test_cindex", ascending=False)

    out = pd.DataFrame({
        "Model": df["model_name"].map(humanize_model_name),
        "Best Test C-index": df["best_test_cindex"].apply(lambda x: fmt_num(x, 4)),
        "Minimum Features within 1\\% of Best": df["min_features_within_1pct_of_best"].fillna("--").astype(str),
        "Minimum Features within 95\\% of Best": df["min_features_within_95pct_of_best"].fillna("--").astype(str),
    })
    return out


def prepare_table_5_risk(
    best_model: str,
    overall_logrank: pd.DataFrame,
    pairwise_logrank: pd.DataFrame,
    pairwise_hr: pd.DataFrame,
) -> pd.DataFrame:
    ov = overall_logrank.loc[
        (overall_logrank["model_name"] == best_model) &
        (overall_logrank["scope"] == "ALL") &
        (overall_logrank["cancer_type"] == "ALL")
    ].copy()

    pl = pairwise_logrank.loc[
        (pairwise_logrank["model_name"] == best_model) &
        (pairwise_logrank["scope"] == "ALL") &
        (pairwise_logrank["cancer_type"] == "ALL")
    ].copy()

    hr = pairwise_hr.loc[
        (pairwise_hr["model_name"] == best_model) &
        (pairwise_hr["scope"] == "ALL") &
        (pairwise_hr["cancer_type"] == "ALL")
    ].copy()

    rows = []
    if not ov.empty:
        rows.append({
            "Comparison": "Overall 3-Group Log-Rank",
            "Log-Rank p": fmt_p(ov.iloc[0]["overall_logrank_pvalue"]),
            "Hazard Ratio (95\\% CI)": "--",
            "HR Status": "--",
        })

    desired_order = [
        ("Low", "Intermediate"),
        ("Intermediate", "High"),
        ("Low", "High"),
    ]

    for g1, g2 in desired_order:
        pl_row = pl.loc[(pl["group_1"] == g1) & (pl["group_2"] == g2)]
        hr_row = hr.loc[(hr["reference_group"] == g1) & (hr["comparison_group"] == g2)]

        logrank_p = fmt_p(pl_row.iloc[0]["logrank_pvalue"]) if not pl_row.empty else "--"

        if not hr_row.empty and hr_row.iloc[0]["hr_status"] == "ok_penalized":
            hr_txt = (
                f"{fmt_num(hr_row.iloc[0]['hazard_ratio'], 3)} "
                f"({fmt_num(hr_row.iloc[0]['ci_lower_95'], 3)} to "
                f"{fmt_num(hr_row.iloc[0]['ci_upper_95'], 3)})"
            )
            hr_status = "Penalized Cox"
        else:
            hr_txt = "Not estimable"
            hr_status = hr_row.iloc[0]["hr_status"] if not hr_row.empty else "--"

        rows.append({
            "Comparison": f"{g1} versus {g2}",
            "Log-Rank p": logrank_p,
            "Hazard Ratio (95\\% CI)": hr_txt,
            "HR Status": hr_status,
        })

    return pd.DataFrame(rows)


def prepare_table_6_stats(
    stat_vs_baseline: pd.DataFrame,
    pooled_benchmark: pd.DataFrame,
) -> pd.DataFrame:
    merge_df = stat_vs_baseline.merge(
        pooled_benchmark[["model_name", "test_cindex"]],
        left_on="comparison_model",
        right_on="model_name",
        how="left"
    )

    merge_df = merge_df.sort_values("delta_cindex_cmp_minus_ref", ascending=False)

    out = pd.DataFrame({
        "Comparison Model": merge_df["comparison_model"].map(humanize_model_name),
        "Test C-index": merge_df["test_cindex"].apply(lambda x: fmt_num(x, 4)),
        "$\\Delta$ C-index versus Clinical Baseline": merge_df["delta_cindex_cmp_minus_ref"].apply(lambda x: fmt_num(x, 4)),
        "95\\% CI": [
            f"{fmt_num(lo, 4)} to {fmt_num(hi, 4)}"
            for lo, hi in zip(merge_df["delta_ci_lower_95"], merge_df["delta_ci_upper_95"])
        ],
        "Raw p": merge_df["p_two_sided_bootstrap"].apply(fmt_p),
        "FDR-Adjusted p": merge_df["p_fdr_bh"].apply(fmt_p),
    })
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg = load_yaml(CONFIG_PATH)

    outputs = cfg["outputs"]
    audit_dir = Path(outputs["audit_dir"])
    tables_dir = Path(outputs["tables_dir"])
    figures_dir = Path(outputs["figures_dir"])
    predictions_dir = Path(outputs["predictions_dir"])

    manuscript_tables_dir = tables_dir / "manuscript"
    manuscript_figures_dir = figures_dir / "manuscript"
    ensure_dir(manuscript_tables_dir)
    ensure_dir(manuscript_figures_dir)

    # -------------------------------------------------------------------------
    # Load required analysis results
    # -------------------------------------------------------------------------
    audit_summary = read_csv_required(audit_dir / "audit_cohort_summary.csv")
    pooled_benchmark = read_csv_required(tables_dir / "pooled_benchmark_results.csv")
    loco_results = read_csv_required(tables_dir / "leave_one_cancer_out_results.csv")
    compactness = read_csv_required(tables_dir / "feature_compactness_results.csv")
    plateau_summary = read_csv_required(tables_dir / "feature_compactness_plateau_summary.csv")
    overall_logrank = read_csv_required(tables_dir / "risk_stratification_overall_logrank.csv")
    pairwise_logrank = read_csv_required(tables_dir / "risk_stratification_pairwise_logrank.csv")
    pairwise_hr = read_csv_required(tables_dir / "risk_stratification_pairwise_hazard_ratios.csv")
    stat_vs_baseline = read_csv_required(tables_dir / "statistical_vs_clinical_baseline_pooled.csv")
    test_predictions_with_risk = read_csv_required(predictions_dir / "test_predictions_with_risk_groups.csv")

    best_model = pooled_benchmark.sort_values("test_cindex", ascending=False).iloc[0]["model_name"]

    # -------------------------------------------------------------------------
    # Prepare and save main tables
    # -------------------------------------------------------------------------
    table_1 = prepare_table_1(audit_summary)
    table_2 = prepare_table_2(pooled_benchmark)
    table_3 = prepare_table_3_loco(loco_results)
    table_4 = prepare_table_4_compactness(plateau_summary)
    table_5 = prepare_table_5_risk(best_model, overall_logrank, pairwise_logrank, pairwise_hr)
    table_6 = prepare_table_6_stats(stat_vs_baseline, pooled_benchmark)

    save_table_bundle(
        df=table_1,
        out_csv=manuscript_tables_dir / "table1_cohort_audit_summary.csv",
        out_tex=manuscript_tables_dir / "table1_cohort_audit_summary.tex",
        caption="Cohort-level audit summary for the three TCGA cancer cohorts included in the leakage-safe survival benchmark.",
        label="tab:cohort_audit_summary",
        note="Dead rate is reported as a percentage. Follow-up is summarized in years. Omics available indicates whether the cohort contained an available omics branch in the present dataset rebuild.",
        table_env="table*",
    )

    save_table_bundle(
        df=table_2,
        out_csv=manuscript_tables_dir / "table2_pooled_benchmark.csv",
        out_tex=manuscript_tables_dir / "table2_pooled_benchmark.tex",
        caption="Pooled validation and test benchmark results across the frozen test set.",
        label="tab:pooled_benchmark",
        note="The clinical baseline is Cox (Clinical Only). Confidence intervals are paired bootstrap 95\\% intervals on the test set.",
        table_env="table*",
    )

    save_table_bundle(
        df=table_3,
        out_csv=manuscript_tables_dir / "table3_leave_one_cancer_out.csv",
        out_tex=manuscript_tables_dir / "table3_leave_one_cancer_out.tex",
        caption="Leave-one-cancer-out test C-index for each model when each cohort is treated as an unseen target cohort.",
        label="tab:loco_results",
        note="Mean denotes the arithmetic mean across available held-out cancer cohorts.",
        table_env="table",
    )

    save_table_bundle(
        df=table_4,
        out_csv=manuscript_tables_dir / "table4_compactness_plateau_summary.csv",
        out_tex=manuscript_tables_dir / "table4_compactness_plateau_summary.tex",
        caption="Compactness plateau summary showing how many top-ranked features are required to approach peak test discrimination.",
        label="tab:compactness_summary",
        note="Minimum features within 1\\% of best identifies the smallest compact feature subset whose test C-index is within 0.01 absolute C-index points of the model-specific best result.",
        table_env="table",
    )

    save_table_bundle(
        df=table_5,
        out_csv=manuscript_tables_dir / "table5_pooled_risk_stratification.csv",
        out_tex=manuscript_tables_dir / "table5_pooled_risk_stratification.tex",
        caption=f"Pooled risk-stratification statistics for the best pooled model: {humanize_model_name(best_model)}.",
        label="tab:pooled_risk_stratification",
        note="Pairwise hazard ratios are reported only when the penalized Cox comparison was stable. Non-estimable comparisons are explicitly marked.",
        table_env="table",
    )

    save_table_bundle(
        df=table_6,
        out_csv=manuscript_tables_dir / "table6_statistical_vs_clinical_baseline.csv",
        out_tex=manuscript_tables_dir / "table6_statistical_vs_clinical_baseline.tex",
        caption="Pooled paired bootstrap statistical comparison of each full-available model versus the Cox clinical baseline.",
        label="tab:statistical_vs_baseline",
        note="FDR-adjusted p-values were computed using the Benjamini-Hochberg procedure.",
        table_env="table*",
    )

    # -------------------------------------------------------------------------
    # Generate manuscript-ready figures
    # -------------------------------------------------------------------------
    fig1 = manuscript_figures_dir / "figure1_study_workflow.png"
    fig2 = manuscript_figures_dir / "figure2_feature_compactness.png"
    fig3 = manuscript_figures_dir / "figure3_pooled_kaplan_meier_risk_stratification.png"

    draw_workflow_figure(fig1)
    draw_compactness_figure(compactness, fig2)
    draw_km_figure(test_predictions_with_risk, best_model, fig3)

    # Supplementary figures
    supp_fig1 = manuscript_figures_dir / "supp_figure1_pooled_benchmark.png"
    supp_fig2 = manuscript_figures_dir / "supp_figure2_leave_one_cancer_out_heatmap.png"
    supp_fig3 = manuscript_figures_dir / "supp_figure3_delta_vs_clinical_baseline.png"

    draw_pooled_benchmark_figure(pooled_benchmark, supp_fig1)
    draw_loco_heatmap(loco_results, supp_fig2)
    draw_delta_vs_baseline_figure(stat_vs_baseline, supp_fig3)

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "best_pooled_model": best_model,
        "main_figures": {
            "figure1_workflow": str(fig1),
            "figure2_compactness": str(fig2),
            "figure3_pooled_kaplan_meier": str(fig3),
        },
        "supplementary_figures": {
            "supp_figure1_pooled_benchmark": str(supp_fig1),
            "supp_figure2_leave_one_cancer_out_heatmap": str(supp_fig2),
            "supp_figure3_delta_vs_clinical_baseline": str(supp_fig3),
        },
        "main_tables": {
            "table1_cohort_audit_summary_csv": str(manuscript_tables_dir / "table1_cohort_audit_summary.csv"),
            "table1_cohort_audit_summary_tex": str(manuscript_tables_dir / "table1_cohort_audit_summary.tex"),
            "table2_pooled_benchmark_csv": str(manuscript_tables_dir / "table2_pooled_benchmark.csv"),
            "table2_pooled_benchmark_tex": str(manuscript_tables_dir / "table2_pooled_benchmark.tex"),
            "table3_leave_one_cancer_out_csv": str(manuscript_tables_dir / "table3_leave_one_cancer_out.csv"),
            "table3_leave_one_cancer_out_tex": str(manuscript_tables_dir / "table3_leave_one_cancer_out.tex"),
            "table4_compactness_plateau_summary_csv": str(manuscript_tables_dir / "table4_compactness_plateau_summary.csv"),
            "table4_compactness_plateau_summary_tex": str(manuscript_tables_dir / "table4_compactness_plateau_summary.tex"),
            "table5_pooled_risk_stratification_csv": str(manuscript_tables_dir / "table5_pooled_risk_stratification.csv"),
            "table5_pooled_risk_stratification_tex": str(manuscript_tables_dir / "table5_pooled_risk_stratification.tex"),
            "table6_statistical_vs_clinical_baseline_csv": str(manuscript_tables_dir / "table6_statistical_vs_clinical_baseline.csv"),
            "table6_statistical_vs_clinical_baseline_tex": str(manuscript_tables_dir / "table6_statistical_vs_clinical_baseline.tex"),
        },
    }
    save_json(manifest, manuscript_tables_dir / "manuscript_artifact_manifest.json")

    print("=" * 80)
    print("MANUSCRIPT FIGURE/TABLE PACKAGING COMPLETE")
    print("=" * 80)
    print(f"Best pooled model: {humanize_model_name(best_model)}")
    print(f"Main figures saved in: {manuscript_figures_dir}")
    print(f"Main tables saved in: {manuscript_tables_dir}")
    print(f"Manifest saved: {manuscript_tables_dir / 'manuscript_artifact_manifest.json'}")


if __name__ == "__main__":
    main()