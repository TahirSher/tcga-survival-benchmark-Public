from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test


CONFIG_PATH = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project\configs\config.yaml")


# =============================================================================
# Same global style as 12_make_figures_tables.py
# =============================================================================

BRIGHT_BLUE = ["#22AFFB", "#50BEFD", "#8CD6FF", "#B2E4FF"]
OCEAN_BREEZE = ["#34C6C1", "#4ADDCE", "#ABE9DB", "#D2F4E5", "#AAE4E8"]
RED_YELLOW_SUNSET = ["#A52222", "#D54536", "#DF6339", "#E29E39", "#F1C461"]

RISK_COLORS = {
    "Low": RED_YELLOW_SUNSET[4],
    "Intermediate": RED_YELLOW_SUNSET[2],
    "High": RED_YELLOW_SUNSET[0],
}

FIG_W = 13.333
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
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def apply_axis_style(ax, use_grid: bool = False, tick_direction: str = "inout") -> None:
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


def fmt_p_scientific(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "--"
    if not np.isfinite(x):
        return "--"
    if x < 1e-4:
        mantissa, exponent = f"{x:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return f"{x:.4f}"


def fmt_p_plain(x: Any) -> str:
    try:
        x = float(x)
    except Exception:
        return "--"
    if not np.isfinite(x):
        return "--"
    if x < 1e-4:
        return "<0.0001"
    return f"{x:.4f}"


def fmt_hr(hr: Any, lo: Any, hi: Any) -> str:
    try:
        return f"{float(hr):.3f} ({float(lo):.3f} to {float(hi):.3f})"
    except Exception:
        return "--"


def safe_int(x: Any) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


# =============================================================================
# Main figure
# =============================================================================

def draw_external_pooled_km_figure(
    pred_df: pd.DataFrame,
    logrank_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    group_counts_df: pd.DataFrame,
    model_name: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    df = pred_df.loc[
        (pred_df["model_name"] == model_name)
        & (pred_df["scope"].fillna("") if "scope" in pred_df.columns else True)
    ].copy()

    # The prediction file normally does not contain scope column; this keeps it robust.
    df = pred_df.loc[pred_df["model_name"] == model_name].copy()

    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["time_days"] = pd.to_numeric(df["time_days"], errors="coerce")
    df = df.dropna(subset=["event", "time_days", "risk_group"]).copy()
    df = df.loc[df["time_days"] > 0].copy()

    if df.empty:
        raise ValueError(f"No external risk-stratified predictions for {model_name}")

    lr = logrank_df.loc[
        (logrank_df["model_name"] == model_name)
        & (logrank_df["scope"] == "ALL_EXTERNAL")
        & (logrank_df["cancer_type"] == "ALL")
    ].iloc[0]

    hr = hr_df.loc[
        (hr_df["model_name"] == model_name)
        & (hr_df["scope"] == "ALL_EXTERNAL")
        & (hr_df["cancer_type"] == "ALL")
        & (hr_df["reference_group"] == "Low")
        & (hr_df["comparison_group"] == "High")
    ].iloc[0]

    counts = group_counts_df.loc[
        (group_counts_df["model_name"] == model_name)
        & (group_counts_df["scope"] == "ALL_EXTERNAL")
        & (group_counts_df["cancer_type"] == "ALL")
    ].copy()

    counts["risk_group"] = pd.Categorical(
        counts["risk_group"],
        categories=["Low", "Intermediate", "High"],
        ordered=True,
    )
    counts = counts.sort_values("risk_group")

    n_total = len(df)
    n_events = int((df["event"] == 1).sum())
    n_censored = int((df["event"] == 0).sum())

    logrank_p = float(lr["overall_logrank_pvalue"])
    hr_txt = fmt_hr(hr["hazard_ratio"], hr["ci_lower_95"], hr["ci_upper_95"])
    hr_p_txt = fmt_p_scientific(hr["p_value"])

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.7, 1.15], hspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")

    kmf = KaplanMeierFitter()
    groups = ["Low", "Intermediate", "High"]

    for grp in groups:
        g = df.loc[df["risk_group"] == grp].copy()
        if g.empty:
            continue

        n_grp = len(g)
        ev_grp = int((g["event"] == 1).sum())
        label = f"{grp} (n={n_grp}, events={ev_grp})"

        kmf.fit(
            durations=g["time_days"],
            event_observed=g["event"],
            label=label,
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=RISK_COLORS.get(grp, BRIGHT_BLUE[1]),
            linewidth=2.4,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Survival Probability", fontweight="bold")
    ax.set_title(
        "External Validation Kaplan?Meier Curves by Frozen TCGA Clinical Cox Risk Group",
        fontweight="bold",
        pad=10,
    )

    annotation = (
        f"External N = {n_total:,}; Events = {n_events:,}; Censored = {n_censored:,}\n"
        rf"Overall log-rank $p$ = {fmt_p_scientific(logrank_p)}"
        "\n"
        rf"High vs Low HR = {hr_txt}; $p$ = {hr_p_txt}"
    )

    ax.text(
        0.985,
        0.965,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#2A2A2A",
            linewidth=0.8,
            alpha=0.96,
        ),
    )

    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.03)

    apply_axis_style(ax, use_grid=False, tick_direction="inout")

    leg = ax.legend(title="Risk Group", loc="lower left")
    if leg is not None:
        leg.get_title().set_fontweight("bold")
        for txt in leg.get_texts():
            txt.set_fontweight("bold")

    # -------------------------------------------------------------------------
    # Compact risk-group table below the plot
    # -------------------------------------------------------------------------
    table_rows = []
    for grp in groups:
        sub = counts.loc[counts["risk_group"].astype(str) == grp]
        if sub.empty:
            table_rows.append([grp, "--", "--", "--"])
            continue

        r = sub.iloc[0]
        n = safe_int(r["n_patients"])
        dead = safe_int(r["n_dead"])
        alive = safe_int(r["n_alive"])
        table_rows.append([grp, f"{n:,}", f"{dead:,}", f"{alive:,}"])

    table = ax_tbl.table(
        cellText=table_rows,
        colLabels=["Risk group", "N", "Events", "Censored"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#2A2A2A")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F2F2F2")
        else:
            cell.set_text_props(weight="bold")
            if col == 0:
                grp = table_rows[row - 1][0]
                cell.set_facecolor(RISK_COLORS.get(grp, "#FFFFFF"))
            else:
                cell.set_facecolor("#FFFFFF")

    ax.set_xlabel("Time (days)", fontweight="bold", labelpad=8)

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = load_yaml(CONFIG_PATH)
    outputs = cfg["outputs"]

    external_tables_dir = Path(outputs["tables_dir"]).parent / "external_validation" / "tables"
    external_predictions_dir = Path(outputs["tables_dir"]).parent / "external_validation" / "predictions"
    figures_dir = Path(outputs["figures_dir"])
    manuscript_figures_dir = figures_dir / "manuscript"
    ensure_dir(manuscript_figures_dir)

    pred_path = external_predictions_dir / "external_predictions_with_tcga_trainval_risk_groups.csv"
    logrank_path = external_tables_dir / "external_risk_stratification_overall_logrank.csv"
    hr_path = external_tables_dir / "external_risk_stratification_pairwise_hazard_ratios.csv"
    group_counts_path = external_tables_dir / "external_risk_stratification_group_counts.csv"

    pred_df = read_csv_required(pred_path)
    logrank_df = read_csv_required(logrank_path)
    hr_df = read_csv_required(hr_path)
    group_counts_df = read_csv_required(group_counts_path)

    out_png = manuscript_figures_dir / "figure_external_pooled_kaplan_meier_risk_stratification.png"
    out_pdf = manuscript_figures_dir / "figure_external_pooled_kaplan_meier_risk_stratification.pdf"

    draw_external_pooled_km_figure(
        pred_df=pred_df,
        logrank_df=logrank_df,
        hr_df=hr_df,
        group_counts_df=group_counts_df,
        model_name="Cox_ClinicalOnly",
        out_png=out_png,
        out_pdf=out_pdf,
    )

    print("=" * 80)
    print("PUBLICATION-READY EXTERNAL KM FIGURE SAVED")
    print("=" * 80)
    print("PNG:", out_png)
    print("PDF:", out_pdf)


if __name__ == "__main__":
    main()
