from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv


CONFIG_PATH = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project\configs\config.yaml")
EXTERNAL_MASTER = Path(
    r"D:\Dr_Abdul_Rehman\TCGA_Paper\External_Validation\clean_external\external_master_stage1_like_usable.csv"
)


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


# -----------------------------------------------------------------------------
# Same feature engineering logic as main TCGA preprocessing
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


def series_or_blank(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series([""] * len(df), index=df.index)


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
        series_or_blank(out, "biospecimen_sample_types") + " | " +
        series_or_blank(out, "biospecimen_specimen_types") + " | " +
        series_or_blank(out, "biospecimen_tissue_types") + " | " +
        series_or_blank(out, "biospecimen_tumor_descriptors")
    )

    out["bio_has_primary_tumor"] = biospecimen_text.apply(lambda x: has_keyword(x, ["primary tumor", "primary"]))
    out["bio_has_normal_sample"] = biospecimen_text.apply(lambda x: has_keyword(x, ["normal"]))
    out["bio_has_blood_normal"] = biospecimen_text.apply(lambda x: has_keyword(x, ["blood derived normal", "peripheral blood"]))
    out["bio_has_solid_tissue_normal"] = biospecimen_text.apply(lambda x: has_keyword(x, ["solid tissue normal"]))
    out["bio_has_ffpe_flag"] = (
        pd.to_numeric(out.get("biospecimen_n_ffpe_true", 0), errors="coerce").fillna(0) > 0
    ).astype(int)

    return out


# -----------------------------------------------------------------------------
# Prediction and metrics
# -----------------------------------------------------------------------------

def structured_y(event: np.ndarray, time_days: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(np.asarray(event).astype(bool), np.asarray(time_days).astype(float))


def compute_cindex(event: np.ndarray, time_days: np.ndarray, risk_score: np.ndarray) -> float:
    event = np.asarray(event).astype(bool)
    time_days = np.asarray(time_days).astype(float)
    risk_score = np.asarray(risk_score).astype(float)

    if len(event) < 5 or event.sum() == 0 or (~event).sum() == 0:
        return np.nan

    try:
        return float(concordance_index_censored(event, time_days, risk_score)[0])
    except Exception:
        return np.nan


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

    train_event_times = train_time_ref[train_event_ref]
    if len(train_event_times) == 0 or len(test_event) == 0:
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
    if int(test_mask.sum()) < 5:
        return {
            "uno_cindex": np.nan,
            "uno_tau_days": tau,
            "n_train_ref_rows": int(len(train_event_ref)),
            "n_test_rows_used": int(test_mask.sum()),
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


def bootstrap_cindex(
    pred_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(pred_df)
    values = []

    if n < 5:
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
        c = compute_cindex(event[idx], time_days[idx], risk_score[idx])
        if np.isfinite(c):
            values.append(c)

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


def subset_matrix(
    X: np.ndarray,
    all_feature_names: list[str],
    selected_feature_names: list[str],
) -> np.ndarray:
    missing = [f for f in selected_feature_names if f not in all_feature_names]
    if missing:
        raise ValueError(f"Selected model features missing from transformed matrix: {missing[:10]}")
    idx = [all_feature_names.index(f) for f in selected_feature_names]
    return X[:, idx]


def raw_predict_scores(
    model_family: str,
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    if model_family == "lifelines_cox":
        df_x = pd.DataFrame(X, columns=feature_names)
        scores = model.predict_partial_hazard(df_x).to_numpy().reshape(-1)
        return scores.astype(float)

    scores = np.asarray(model.predict(X)).reshape(-1)
    return scores.astype(float)


def infer_matrix_type(model_family: str) -> str:
    if model_family in {"gb", "rsf"}:
        return "tree"
    return "linear"


# -----------------------------------------------------------------------------
# Risk stratification
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


def overall_logrank_three_group(df: pd.DataFrame) -> dict[str, float]:
    if df["risk_group"].nunique() < 2:
        return {"overall_logrank_statistic": np.nan, "overall_logrank_pvalue": np.nan}

    try:
        result = multivariate_logrank_test(
            event_durations=df["time_days"],
            groups=df["risk_group"],
            event_observed=df["event"],
        )
        return {
            "overall_logrank_statistic": float(result.test_statistic),
            "overall_logrank_pvalue": float(result.p_value),
        }
    except Exception:
        return {"overall_logrank_statistic": np.nan, "overall_logrank_pvalue": np.nan}


def pairwise_logrank(df: pd.DataFrame, g1: str, g2: str) -> dict[str, float | str]:
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

    try:
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
    except Exception:
        return {
            "group_1": g1,
            "group_2": g2,
            "n_group_1": int(len(a)),
            "n_group_2": int(len(b)),
            "logrank_statistic": np.nan,
            "logrank_pvalue": np.nan,
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
            "hr_status": "empty_comparison",
        }

    ref_df = temp.loc[temp["risk_group"] == g_ref]
    cmp_df = temp.loc[temp["risk_group"] == g_cmp]

    n_ref = int(len(ref_df))
    n_cmp = int(len(cmp_df))
    n_events_ref = int((ref_df["event"] == 1).sum())
    n_events_cmp = int((cmp_df["event"] == 1).sum())
    n_nonevents_ref = int((ref_df["event"] == 0).sum())
    n_nonevents_cmp = int((cmp_df["event"] == 0).sum())

    base = {
        "reference_group": g_ref,
        "comparison_group": g_cmp,
        "n_rows": int(len(temp)),
        "n_ref": n_ref,
        "n_cmp": n_cmp,
        "n_events_ref": n_events_ref,
        "n_events_cmp": n_events_cmp,
    }

    if n_ref < 10 or n_cmp < 10:
        return {**base, "hazard_ratio": np.nan, "ci_lower_95": np.nan, "ci_upper_95": np.nan, "p_value": np.nan, "hr_status": "skipped_small_group"}
    if n_events_ref < 3 or n_events_cmp < 3:
        return {**base, "hazard_ratio": np.nan, "ci_lower_95": np.nan, "ci_upper_95": np.nan, "p_value": np.nan, "hr_status": "skipped_too_few_events"}
    if n_nonevents_ref < 3 or n_nonevents_cmp < 3:
        return {**base, "hazard_ratio": np.nan, "ci_lower_95": np.nan, "ci_upper_95": np.nan, "p_value": np.nan, "hr_status": "skipped_too_few_nonevents"}

    temp["indicator"] = (temp["risk_group"] == g_cmp).astype(int)
    fit_df = temp[["time_days", "event", "indicator"]].copy()

    try:
        cph = CoxPHFitter(penalizer=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cph.fit(fit_df, duration_col="time_days", event_col="event", show_progress=False)

        return {
            **base,
            "hazard_ratio": float(np.exp(cph.params_["indicator"])),
            "ci_lower_95": float(np.exp(cph.confidence_intervals_.loc["indicator"].iloc[0])),
            "ci_upper_95": float(np.exp(cph.confidence_intervals_.loc["indicator"].iloc[1])),
            "p_value": float(cph.summary.loc["indicator", "p"]),
            "hr_status": "penalized_cox",
        }
    except Exception as e:
        return {
            **base,
            "hazard_ratio": np.nan,
            "ci_lower_95": np.nan,
            "ci_upper_95": np.nan,
            "p_value": np.nan,
            "hr_status": f"failed_{type(e).__name__}",
        }


def summarize_group_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("risk_group", dropna=False)
        .agg(
            n_patients=("patient_id", "count"),
            n_dead=("event", lambda s: int((s == 1).sum())),
            n_alive=("event", lambda s: int((s == 0).sum())),
            median_time_days=("time_days", "median"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )


def km_curve_three_groups(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty or df["risk_group"].nunique() < 2:
        return

    plt.figure(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    for grp in ["Low", "Intermediate", "High"]:
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

    tcga_root = Path(r"D:\Dr_Abdul_Rehman\TCGA_Paper\TCGA_project")
    outputs_dir = tcga_root / "outputs"
    processed_dir = outputs_dir / "processed"
    models_dir = outputs_dir / "models"
    predictions_dir = outputs_dir / "predictions"

    external_dir = outputs_dir / "external_validation"
    external_tables_dir = external_dir / "tables"
    external_predictions_dir = external_dir / "predictions"
    external_figures_dir = external_dir / "figures"
    external_audit_dir = external_dir / "audit"

    for p in [external_dir, external_tables_dir, external_predictions_dir, external_figures_dir, external_audit_dir]:
        ensure_dir(p)

    bundle_path = processed_dir / "preprocessed_data_bundle.joblib"
    trainval_predictions_path = predictions_dir / "trainval_predictions.csv"

    if not EXTERNAL_MASTER.exists():
        raise FileNotFoundError(f"Missing external master: {EXTERNAL_MASTER}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing preprocessing bundle: {bundle_path}")
    if not trainval_predictions_path.exists():
        raise FileNotFoundError(f"Missing trainval predictions: {trainval_predictions_path}")

    # Load external usable table and apply same feature engineering
    external_raw = pd.read_csv(EXTERNAL_MASTER, low_memory=False)
    external_raw["event"] = pd.to_numeric(external_raw["event"], errors="coerce")
    external_raw["time_to_event_days_raw"] = pd.to_numeric(external_raw["time_to_event_days_raw"], errors="coerce")
    external_raw = external_raw.dropna(subset=["patient_id", "cancer_type", "event", "time_to_event_days_raw"]).copy()
    external_raw = external_raw.loc[external_raw["time_to_event_days_raw"] > 0].copy()
    external_df = engineer_features(external_raw)

    # Load frozen TCGA preprocessing bundle
    bundle = joblib.load(bundle_path)
    linear_feature_names = list(bundle["linear_feature_names"])
    tree_feature_names = list(bundle["tree_feature_names"])

    X_external_linear = bundle["linear_pipeline"].transform(external_df)
    X_external_tree = bundle["tree_pipeline"].transform(external_df)

    # TCGA train+val reference for Uno C-index
    train_event_ref = np.concatenate([
        np.asarray(bundle["splits"]["train"]["event"], dtype=float),
        np.asarray(bundle["splits"]["val"]["event"], dtype=float),
    ])
    train_time_ref = np.concatenate([
        np.asarray(bundle["splits"]["train"]["time_days"], dtype=float),
        np.asarray(bundle["splits"]["val"]["time_days"], dtype=float),
    ])

    model_names = [
        "Cox_ClinicalOnly",
        "Cox_FullAvailable",
        "ElasticNetCox_FullAvailable",
        "RandomSurvivalForest_FullAvailable",
        "GradientBoostingSurvival_FullAvailable",
        "SurvivalSVM_FullAvailable",
    ]

    prediction_rows = []
    skipped_rows = []

    for model_name in model_names:
        artifact_path = models_dir / f"{model_name}.joblib"
        if not artifact_path.exists():
            skipped_rows.append({"model_name": model_name, "reason": "missing_model_artifact"})
            continue

        try:
            artifact = joblib.load(artifact_path)
            model_family = artifact["model_family"]
            feature_names = list(artifact["feature_names"])
            risk_sign = float(artifact["risk_sign_applied"])
            final_model = artifact["final_model"]

            matrix_type = infer_matrix_type(model_family)
            if matrix_type == "tree":
                all_names = tree_feature_names
                X_all = X_external_tree
            else:
                all_names = linear_feature_names
                X_all = X_external_linear

            X_model = subset_matrix(X_all, all_names, feature_names)
            scores = risk_sign * raw_predict_scores(model_family, final_model, X_model, feature_names)

            for i, score in enumerate(scores):
                prediction_rows.append({
                    "model_name": model_name,
                    "model_family": model_family,
                    "feature_set": artifact.get("feature_set", ""),
                    "matrix_type": matrix_type,
                    "patient_id": str(external_df.iloc[i]["patient_id"]),
                    "external_dataset": str(external_df.iloc[i]["external_dataset"]),
                    "cancer_type": str(external_df.iloc[i]["cancer_type"]),
                    "event": float(external_df.iloc[i]["event"]),
                    "time_days": float(external_df.iloc[i]["time_to_event_days_raw"]),
                    "risk_score": float(score),
                })

            print(f"[OK] External prediction complete: {model_name}")

        except Exception as e:
            skipped_rows.append({"model_name": model_name, "reason": f"{type(e).__name__}: {str(e)}"})
            print(f"[SKIP] {model_name}: {type(e).__name__}: {e}")

    pred_df = pd.DataFrame(prediction_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    pred_path = external_predictions_dir / "external_predictions_frozen_tcga_models.csv"
    skipped_path = external_tables_dir / "external_validation_skipped_models.csv"

    pred_df.to_csv(pred_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    # C-index summaries
    overall_rows = []
    by_dataset_rows = []
    by_cancer_rows = []

    for model_name, g in pred_df.groupby("model_name", dropna=False):
        event = g["event"].to_numpy(dtype=float)
        time_days = g["time_days"].to_numpy(dtype=float)
        risk_score = g["risk_score"].to_numpy(dtype=float)

        cindex = compute_cindex(event, time_days, risk_score)
        uno = compute_uno_cindex(train_event_ref, train_time_ref, event, time_days, risk_score)
        boot = bootstrap_cindex(g, n_bootstrap=1000, seed=42)

        overall_rows.append({
            "model_name": model_name,
            "scope": "ALL_EXTERNAL",
            "n_patients": int(len(g)),
            "n_events": int((g["event"] == 1).sum()),
            "n_censored": int((g["event"] == 0).sum()),
            "external_cindex": cindex,
            **uno,
            **boot,
        })

        for (dataset, cancer_type), gg in g.groupby(["external_dataset", "cancer_type"], dropna=False):
            event2 = gg["event"].to_numpy(dtype=float)
            time2 = gg["time_days"].to_numpy(dtype=float)
            score2 = gg["risk_score"].to_numpy(dtype=float)

            row = {
                "model_name": model_name,
                "external_dataset": dataset,
                "cancer_type": cancer_type,
                "n_patients": int(len(gg)),
                "n_events": int((gg["event"] == 1).sum()),
                "n_censored": int((gg["event"] == 0).sum()),
                "external_cindex": compute_cindex(event2, time2, score2),
            }
            row.update(compute_uno_cindex(train_event_ref, train_time_ref, event2, time2, score2))
            by_dataset_rows.append(row)

        for cancer_type, gg in g.groupby("cancer_type", dropna=False):
            event2 = gg["event"].to_numpy(dtype=float)
            time2 = gg["time_days"].to_numpy(dtype=float)
            score2 = gg["risk_score"].to_numpy(dtype=float)

            row = {
                "model_name": model_name,
                "cancer_type": cancer_type,
                "n_patients": int(len(gg)),
                "n_events": int((gg["event"] == 1).sum()),
                "n_censored": int((gg["event"] == 0).sum()),
                "external_cindex": compute_cindex(event2, time2, score2),
            }
            row.update(compute_uno_cindex(train_event_ref, train_time_ref, event2, time2, score2))
            by_cancer_rows.append(row)

    overall_df = pd.DataFrame(overall_rows).sort_values("external_cindex", ascending=False)
    by_dataset_df = pd.DataFrame(by_dataset_rows).sort_values(["model_name", "external_dataset"])
    by_cancer_df = pd.DataFrame(by_cancer_rows).sort_values(["model_name", "cancer_type"])

    overall_path = external_tables_dir / "external_validation_overall_cindex.csv"
    by_dataset_path = external_tables_dir / "external_validation_by_dataset_cindex.csv"
    by_cancer_path = external_tables_dir / "external_validation_by_cancer_cindex.csv"

    overall_df.to_csv(overall_path, index=False)
    by_dataset_df.to_csv(by_dataset_path, index=False)
    by_cancer_df.to_csv(by_cancer_path, index=False)

    # Risk stratification using TCGA train+validation thresholds
    trainval_predictions = pd.read_csv(trainval_predictions_path)
    risk_pred_rows = []
    threshold_rows = []
    overall_logrank_rows = []
    pairwise_logrank_rows = []
    hr_rows = []
    group_count_rows = []

    for model_name, g in pred_df.groupby("model_name", dropna=False):
        trv = trainval_predictions.loc[trainval_predictions["model_name"] == model_name].copy()
        if trv.empty:
            continue

        q1, q2 = training_quantile_thresholds(trv)
        gg = g.copy()
        gg["risk_group"] = assign_three_risk_groups(gg["risk_score"], q1=q1, q2=q2)

        threshold_rows.append({
            "model_name": model_name,
            "tcga_trainval_q1_threshold": q1,
            "tcga_trainval_q2_threshold": q2,
            "n_tcga_trainval": int(len(trv)),
            "n_external": int(len(gg)),
        })

        risk_pred_rows.append(gg)

        scopes = [("ALL_EXTERNAL", "ALL", gg)]
        for (dataset, cancer_type), sub in gg.groupby(["external_dataset", "cancer_type"], dropna=False):
            scopes.append((str(dataset), str(cancer_type), sub.copy()))

        for scope, cancer_type, sub in scopes:
            overall_stats = overall_logrank_three_group(sub)
            overall_logrank_rows.append({
                "model_name": model_name,
                "scope": scope,
                "cancer_type": cancer_type,
                **overall_stats,
            })

            gc = summarize_group_counts(sub)
            gc["model_name"] = model_name
            gc["scope"] = scope
            gc["cancer_type"] = cancer_type
            group_count_rows.append(gc)

            pair_specs = [("Low", "Intermediate"), ("Intermediate", "High"), ("Low", "High")]
            for g1, g2 in pair_specs:
                pr = pairwise_logrank(sub, g1, g2)
                pr.update({"model_name": model_name, "scope": scope, "cancer_type": cancer_type})
                pairwise_logrank_rows.append(pr)

                hr = pairwise_hazard_ratio(sub, g1, g2)
                hr.update({"model_name": model_name, "scope": scope, "cancer_type": cancer_type})
                hr_rows.append(hr)

            safe_scope = re.sub(r"[^A-Za-z0-9_]+", "_", f"{scope}_{cancer_type}")
            km_curve_three_groups(
                df=sub,
                title=f"{model_name} external risk stratification: {scope} {cancer_type}",
                output_path=external_figures_dir / f"km_external_{model_name}_{safe_scope}.png",
            )

    risk_predictions_df = pd.concat(risk_pred_rows, axis=0, ignore_index=True) if risk_pred_rows else pd.DataFrame()
    thresholds_df = pd.DataFrame(threshold_rows)
    overall_logrank_df = pd.DataFrame(overall_logrank_rows)
    pairwise_logrank_df = pd.DataFrame(pairwise_logrank_rows)
    hr_df = pd.DataFrame(hr_rows)
    group_counts_df = pd.concat(group_count_rows, axis=0, ignore_index=True) if group_count_rows else pd.DataFrame()

    risk_predictions_path = external_predictions_dir / "external_predictions_with_tcga_trainval_risk_groups.csv"
    thresholds_path = external_tables_dir / "external_risk_group_thresholds_from_tcga_trainval.csv"
    overall_logrank_path = external_tables_dir / "external_risk_stratification_overall_logrank.csv"
    pairwise_logrank_path = external_tables_dir / "external_risk_stratification_pairwise_logrank.csv"
    hr_path = external_tables_dir / "external_risk_stratification_pairwise_hazard_ratios.csv"
    group_counts_path = external_tables_dir / "external_risk_stratification_group_counts.csv"

    risk_predictions_df.to_csv(risk_predictions_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    overall_logrank_df.to_csv(overall_logrank_path, index=False)
    pairwise_logrank_df.to_csv(pairwise_logrank_path, index=False)
    hr_df.to_csv(hr_path, index=False)
    group_counts_df.to_csv(group_counts_path, index=False)

    # Compact manuscript-style external table
    manuscript_rows = []
    for _, row in overall_df.iterrows():
        manuscript_rows.append({
            "Model": row["model_name"],
            "External N": int(row["n_patients"]),
            "Events": int(row["n_events"]),
            "Harrell C-index": round(float(row["external_cindex"]), 4) if pd.notna(row["external_cindex"]) else np.nan,
            "Uno C-index": round(float(row["uno_cindex"]), 4) if pd.notna(row["uno_cindex"]) else np.nan,
            "95% CI": (
                f"{row['ci_lower_95']:.4f} to {row['ci_upper_95']:.4f}"
                if pd.notna(row["ci_lower_95"]) and pd.notna(row["ci_upper_95"])
                else ""
            ),
        })

    manuscript_external_path = external_tables_dir / "manuscript_external_validation_summary.csv"
    pd.DataFrame(manuscript_rows).to_csv(manuscript_external_path, index=False)

    summary = {
        "external_master": str(EXTERNAL_MASTER),
        "bundle_path": str(bundle_path),
        "n_external_rows": int(len(external_df)),
        "n_models_predicted": int(pred_df["model_name"].nunique()) if not pred_df.empty else 0,
        "primary_interpretation": "Use Cox_ClinicalOnly as primary external validation; full-available models are secondary sensitivity because external cohorts lack true TCGA biospecimen and omics-availability descriptors.",
        "outputs": {
            "predictions": str(pred_path),
            "overall_cindex": str(overall_path),
            "by_dataset_cindex": str(by_dataset_path),
            "by_cancer_cindex": str(by_cancer_path),
            "risk_predictions": str(risk_predictions_path),
            "risk_thresholds": str(thresholds_path),
            "overall_logrank": str(overall_logrank_path),
            "pairwise_logrank": str(pairwise_logrank_path),
            "pairwise_hazard_ratios": str(hr_path),
            "group_counts": str(group_counts_path),
            "manuscript_external_summary": str(manuscript_external_path),
            "skipped_models": str(skipped_path),
        },
    }

    save_json(summary, external_audit_dir / "external_validation_frozen_tcga_summary.json")

    print("=" * 80)
    print("FROZEN TCGA EXTERNAL VALIDATION COMPLETE")
    print("=" * 80)
    print("\nOverall external C-index:")
    print(overall_df[[
        "model_name",
        "n_patients",
        "n_events",
        "n_censored",
        "external_cindex",
        "uno_cindex",
        "ci_lower_95",
        "ci_upper_95",
    ]].to_string(index=False))

    print("\nBy dataset external C-index:")
    print(by_dataset_df[[
        "model_name",
        "external_dataset",
        "cancer_type",
        "n_patients",
        "n_events",
        "n_censored",
        "external_cindex",
        "uno_cindex",
    ]].to_string(index=False))

    print("\nSaved outputs:")
    print(" -", pred_path)
    print(" -", overall_path)
    print(" -", by_dataset_path)
    print(" -", manuscript_external_path)
    print(" -", external_audit_dir / "external_validation_frozen_tcga_summary.json")


if __name__ == "__main__":
    main()
