#!/usr/bin/env python


from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn import __version__ as sklearn_version
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 20260831
OUTCOME = "预测值"
OUTER_SPLITS = 5
OUTER_REPEATS = 10
INNER_SPLITS = 5
MAX_ITER = 10000
COEF_TOL = 1e-7
C_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
L1_GRID = (0.2, 0.5, 0.8, 1.0)

FEATURE_DEFINITIONS = [
    ("Male sex", "Is_male", "0/1", "direct"),
    ("Weight (kg)", "Weight", "kg", "direct"),
    ("Preoperative CRP", "Preoperative CRP", "source unit", "direct"),
    ("Preoperative neutrophils", "Preoperative Neutrophils", "source unit", "direct"),
    ("Driving pressure (OLV-30 min)", "Driving Pressure (OLV-30min)", "cmH2O", "direct; checked against Pplat - PEEP"),
    ("PEEP (OLV-30 min)", "PEEP (OLV-30min)", "cmH2O", "direct"),
    ("PaCO2 (OLV-30 min)", "PaCO2 (OLV-30min)", "mmHg", "direct"),
    ("Blood base excess (OLV-30 min)", "Base Excess of Blood (OLV-30min)\t", "mmol/L", "direct"),
    ("Heart rate (OLV-30 min)", "Heart Rate (OLV-30min)", "beats/min", "direct"),
    (
        "Ti/Ttot (OLV-30 min)",
        "Ti/Ttot (OLV-30min)",
        "ratio",
        "direct; checked against Inspiratory Time × RR / 60",
    ),
]
FEATURES = [row[0] for row in FEATURE_DEFINITIONS]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "data" / "original_data_p6e.xlsx")
    parser.add_argument("--output", type=Path, default=here / "results" / "primary")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples of mean repeated OOF predictions")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path):
    raw = pd.read_excel(path)
    required = [row[1] for row in FEATURE_DEFINITIONS] + [
        "Inspiratory Time (OLV-30min)", "RR (OLV-30min)", "Pplat (OLV-30min)", OUTCOME
    ]
    required = sorted(set(required))
    missing_columns = [c for c in required if c not in raw.columns]
    if missing_columns:
        raise KeyError(f"Required source columns not found: {missing_columns}")

    x = pd.DataFrame(index=raw.index)
    # All ten model predictors, including driving pressure and Ti/Ttot, are
    # read directly from their existing columns in the source workbook.
    for display, source, _, _ in FEATURE_DEFINITIONS:
        x[display] = pd.to_numeric(raw[source], errors="coerce")

    # Reconstruct Ti/Ttot solely as an independent data-quality check. The
    # reconstructed value never replaces the recorded source value.
    ti = pd.to_numeric(raw["Inspiratory Time (OLV-30min)"], errors="coerce")
    rr = pd.to_numeric(raw["RR (OLV-30min)"], errors="coerce")
    reconstructed_ti_ttot = ti * rr / 60.0
    ti_ttot_formula_error = (
        x["Ti/Ttot (OLV-30 min)"] - reconstructed_ti_ttot
    ).abs()
    ti_ttot_mismatch = ti_ttot_formula_error > 1e-10
    if ti_ttot_mismatch.any():
        mismatch_rows = (ti_ttot_mismatch[ti_ttot_mismatch].index + 2).tolist()
        raise ValueError(
            "Recorded Ti/Ttot (OLV-30min) is inconsistent with "
            "Inspiratory Time × RR / 60 in Excel row(s): "
            f"{mismatch_rows[:20]}"
        )

    y = pd.to_numeric(raw[OUTCOME], errors="raise").astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("Outcome must contain only 0 and 1")

    # Driving pressure is likewise read directly from the workbook. Pplat -
    # PEEP is calculated only to verify the recorded driving-pressure column.
    pplat = pd.to_numeric(raw["Pplat (OLV-30min)"], errors="coerce")
    driving_pressure_formula_error = (
        x["Driving pressure (OLV-30 min)"]
        - (pplat - x["PEEP (OLV-30 min)"])
    ).abs()
    driving_pressure_mismatch = driving_pressure_formula_error > 1e-10
    if driving_pressure_mismatch.any():
        mismatch_rows = (driving_pressure_mismatch[driving_pressure_mismatch].index + 2).tolist()
        raise ValueError(
            "Recorded Driving Pressure (OLV-30min) is inconsistent with "
            "Pplat - PEEP in Excel row(s): "
            f"{mismatch_rows[:20]}"
        )

    checks = {
        "n_patients": int(len(y)),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "outcome_column": OUTCOME,
        "driving_pressure_source_column": "Driving Pressure (OLV-30min)",
        "driving_pressure_verification_formula": "Pplat (OLV-30min) - PEEP (OLV-30min)",
        "max_absolute_driving_pressure_formula_error": float(driving_pressure_formula_error.max(skipna=True)),
        "n_rows_driving_pressure_formula_checked": int(driving_pressure_formula_error.notna().sum()),
        "n_rows_driving_pressure_formula_mismatch": int(driving_pressure_mismatch.sum()),
        "ti_ttot_source_column": "Ti/Ttot (OLV-30min)",
        "ti_ttot_verification_formula": "Inspiratory Time (OLV-30min) * RR (OLV-30min) / 60",
        "n_rows_ti_ttot_observed": int(x["Ti/Ttot (OLV-30 min)"].notna().sum()),
        "max_absolute_ti_ttot_formula_error": float(ti_ttot_formula_error.max(skipna=True)),
        "n_rows_ti_ttot_formula_checked": int(ti_ttot_formula_error.notna().sum()),
        "n_rows_ti_ttot_formula_mismatch": int(ti_ttot_mismatch.sum()),
    }
    return raw, x[FEATURES], y, checks


def make_model(penalty: str, c: float, l1_ratio: float | None, seed: int):
    solver = "lbfgs" if penalty == "l2" else "saga"
    kwargs = dict(penalty=penalty, C=c, solver=solver, max_iter=MAX_ITER, random_state=seed)
    if penalty == "elasticnet":
        kwargs["l1_ratio"] = l1_ratio
    return LogisticRegression(**kwargs)


def make_pipeline(penalty: str, c: float, l1_ratio: float | None, seed: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", make_model(penalty, c, l1_ratio, seed)),
    ])


def prepare_inner(x: pd.DataFrame, y: pd.Series, seed: int):
    splits = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    prepared = []
    for inner_fold, (tr, va) in enumerate(splits.split(x, y), start=1):
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        a = prep.fit_transform(x.iloc[tr])
        b = prep.transform(x.iloc[va])
        prepared.append((inner_fold, a, y.iloc[tr].to_numpy(), b, y.iloc[va].to_numpy()))
    return prepared


def tune(prepared, model_name: str, penalty: str, seed: int):
    rows = []
    l1_values = L1_GRID if penalty == "elasticnet" else (None,)
    for c in C_GRID:
        for l1_ratio in l1_values:
            losses, counts = [], []
            for inner_fold, a, ya, b, yb in prepared:
                model = make_model(penalty, c, l1_ratio, seed + inner_fold)
                model.fit(a, ya)
                prob = model.predict_proba(b)[:, 1]
                losses.append(log_loss(yb, prob, labels=[0, 1]))
                counts.append(np.sum(np.abs(model.coef_.ravel()) > COEF_TOL))
            rows.append({
                "model": model_name,
                "C": c,
                "l1_ratio": l1_ratio,
                "mean_inner_log_loss": float(np.mean(losses)),
                "se_inner_log_loss": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                "mean_n_selected": float(np.mean(counts)),
            })
    grid = pd.DataFrame(rows).sort_values(
        ["mean_inner_log_loss", "mean_n_selected", "C"], ascending=[True, True, True]
    ).reset_index(drop=True)
    chosen = grid.iloc[0].to_dict()
    return chosen, grid


def calibration_coefficients(y: np.ndarray, p: np.ndarray):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))

    def objective(beta):
        q = expit(beta[0] + beta[1] * z)
        return -np.sum(y * np.log(np.clip(q, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1)))

    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return float(fit.x[0]), float(fit.x[1])


def performance(y: np.ndarray, p: np.ndarray):
    intercept, slope = calibration_coefficients(y, p)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Log loss": float(log_loss(y, p, labels=[0, 1])),
        "Calibration intercept": intercept,
        "Calibration slope": slope,
    }


def run_nested_cv(x: pd.DataFrame, y: pd.Series):
    model_specs = [
        ("ridge", "l2"),
        ("elastic_net", "elasticnet"),
        ("lasso", "l1"),
    ]
    predictions, hyper_rows, tuning_frames, coefficient_rows = [], [], [], []
    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED
    )
    for outer_index, (tr, te) in enumerate(outer.split(x, y)):
        repeat = outer_index // OUTER_SPLITS + 1
        fold = outer_index % OUTER_SPLITS + 1
        seed = SEED + repeat * 1000 + fold
        xtr, ytr = x.iloc[tr], y.iloc[tr]
        prepared = prepare_inner(xtr, ytr, seed)
        for model_name, penalty in model_specs:
            chosen, grid = tune(prepared, model_name, penalty, seed)
            grid.insert(0, "outer_fold", fold)
            grid.insert(0, "outer_repeat", repeat)
            tuning_frames.append(grid)
            c = float(chosen["C"])
            l1_ratio = None if pd.isna(chosen["l1_ratio"]) else float(chosen["l1_ratio"])
            pipe = make_pipeline(penalty, c, l1_ratio, seed)
            pipe.fit(xtr, ytr)
            prob = pipe.predict_proba(x.iloc[te])[:, 1]
            coef = pipe.named_steps["model"].coef_.ravel()
            hyper_rows.append({
                "model": model_name,
                "outer_repeat": repeat,
                "outer_fold": fold,
                "C": c,
                "l1_ratio": l1_ratio,
                "n_selected": int(np.sum(np.abs(coef) > COEF_TOL)),
                "inner_mean_log_loss": float(chosen["mean_inner_log_loss"]),
                "inner_se_log_loss": float(chosen["se_inner_log_loss"]),
            })
            for feature, value in zip(FEATURES, coef):
                coefficient_rows.append({
                    "model": model_name,
                    "outer_repeat": repeat,
                    "outer_fold": fold,
                    "feature": feature,
                    "selected": int(abs(value) > COEF_TOL),
                    "standardized_coefficient": float(value),
                })
            for idx, p in zip(te, prob):
                predictions.append({
                    "model": model_name,
                    "outer_repeat": repeat,
                    "outer_fold": fold,
                    "row_index_zero_based": int(idx),
                    "row_number_excel_including_header": int(idx + 2),
                    "observed": int(y.iloc[idx]),
                    "predicted_probability": float(p),
                })
        print(f"Outer repeat {repeat}/{OUTER_REPEATS}, fold {fold}/{OUTER_SPLITS} complete")
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(hyper_rows),
        pd.concat(tuning_frames, ignore_index=True),
        pd.DataFrame(coefficient_rows),
    )


def summarize_predictions(predictions: pd.DataFrame):
    repeat_rows = []
    for (model, repeat), group in predictions.groupby(["model", "outer_repeat"]):
        repeat_rows.append({"model": model, "outer_repeat": repeat, **performance(group.observed.to_numpy(), group.predicted_probability.to_numpy())})
    by_repeat = pd.DataFrame(repeat_rows)

    mean_oof = (
        predictions.groupby(["model", "row_index_zero_based", "row_number_excel_including_header"], as_index=False)
        .agg(observed=("observed", "first"), predicted_probability=("predicted_probability", "mean"))
    )
    summary_rows = []
    for model, group in by_repeat.groupby("model"):
        pooled = mean_oof.loc[mean_oof.model == model]
        pooled_perf = performance(pooled.observed.to_numpy(), pooled.predicted_probability.to_numpy())
        for metric in ["AUC", "AUPRC", "Brier", "Log loss", "Calibration intercept", "Calibration slope"]:
            values = group[metric]
            summary_rows.append({
                "model": model,
                "metric": metric,
                "mean_across_10_repeats": float(values.mean()),
                "sd_across_10_repeats": float(values.std(ddof=1)),
                "resampling_p2_5": float(values.quantile(0.025)),
                "resampling_p97_5": float(values.quantile(0.975)),
                "metric_from_mean_repeated_oof_probability": pooled_perf[metric],
            })
    return pd.DataFrame(summary_rows), by_repeat, mean_oof


def fold_performance(predictions: pd.DataFrame):
    rows = []
    for (model, repeat, fold), group in predictions.groupby(["model", "outer_repeat", "outer_fold"]):
        rows.append({"model": model, "outer_repeat": repeat, "outer_fold": fold, "n_test": len(group), "events_test": int(group.observed.sum()), **performance(group.observed.to_numpy(), group.predicted_probability.to_numpy())})
    return pd.DataFrame(rows)


def selection_stability(coef_rows: pd.DataFrame):
    return (
        coef_rows.groupby(["model", "feature"], as_index=False)
        .agg(
            selection_frequency=("selected", "mean"),
            median_standardized_coefficient=("standardized_coefficient", "median"),
            coefficient_p2_5=("standardized_coefficient", lambda s: float(np.quantile(s, 0.025))),
            coefficient_p97_5=("standardized_coefficient", lambda s: float(np.quantile(s, 0.975))),
        )
        .sort_values(["model", "selection_frequency"], ascending=[True, False])
    )


def candidate_tables(x: pd.DataFrame, y: pd.Series):
    definition_rows, descriptive_rows = [], []
    for display, source, unit, derivation in FEATURE_DEFINITIONS:
        definition_rows.append({
            "feature": display,
            "source_column_or_columns": source,
            "unit": unit,
            "derivation": derivation,
            "missing_n": int(x[display].isna().sum()),
            "missing_percent": float(x[display].isna().mean() * 100),
        })
        for value, group_name in [(None, "Overall"), (0, "No impairment"), (1, "Impairment")]:
            s = x[display] if value is None else x.loc[y == value, display]
            descriptive_rows.append({
                "feature": display,
                "group": group_name,
                "n_observed": int(s.notna().sum()),
                "missing_n": int(s.isna().sum()),
                "mean": float(s.mean()),
                "SD": float(s.std(ddof=1)),
                "median": float(s.median()),
                "Q1": float(s.quantile(0.25)),
                "Q3": float(s.quantile(0.75)),
                "minimum": float(s.min()),
                "maximum": float(s.max()),
            })
    return pd.DataFrame(definition_rows), pd.DataFrame(descriptive_rows)


def vif_table(x: pd.DataFrame):
    filled = SimpleImputer(strategy="median").fit_transform(x)
    rows = []
    for j, feature in enumerate(x.columns):
        target = filled[:, j]
        others = np.delete(filled, j, axis=1)
        fit = np.column_stack([np.ones(len(others)), others])
        beta = np.linalg.lstsq(fit, target, rcond=None)[0]
        predicted = fit @ beta
        ss_total = np.sum((target - target.mean()) ** 2)
        r2 = 0.0 if ss_total == 0 else 1 - np.sum((target - predicted) ** 2) / ss_total
        rows.append({"feature": feature, "R_squared_against_other_candidates": r2, "VIF": np.inf if r2 >= 1 else 1 / (1 - r2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def fit_full_cohort_models(x: pd.DataFrame, y: pd.Series):
    model_specs = [("ridge", "l2"), ("elastic_net", "elasticnet"), ("lasso", "l1")]
    prepared = prepare_inner(x, y, SEED + 900000)
    grids, coefficients, choices = [], [], []
    for model_name, penalty in model_specs:
        chosen, grid = tune(prepared, model_name, penalty, SEED + 900000)
        grids.append(grid)
        c = float(chosen["C"])
        l1_ratio = None if pd.isna(chosen["l1_ratio"]) else float(chosen["l1_ratio"])
        pipe = make_pipeline(penalty, c, l1_ratio, SEED + 900000)
        pipe.fit(x, y)
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        model = pipe.named_steps["model"]
        choices.append({"model": model_name, "C": c, "l1_ratio": l1_ratio, "n_selected": int(np.sum(np.abs(model.coef_.ravel()) > COEF_TOL))})
        standardized_intercept = float(model.intercept_[0])
        original_intercept = standardized_intercept - float(np.sum(model.coef_.ravel() * scaler.mean_ / scaler.scale_))
        coefficients.append({
            "model": model_name,
            "term": "Intercept",
            "selected": 1,
            "imputation_median": np.nan,
            "standardization_mean_after_imputation": np.nan,
            "standardization_SD_after_imputation": np.nan,
            "standardized_coefficient": standardized_intercept,
            "OR_per_1_SD": np.nan,
            "coefficient_per_original_unit": original_intercept,
            "OR_per_original_unit": np.nan,
        })
        for feature, median, mean, scale, beta in zip(FEATURES, imputer.statistics_, scaler.mean_, scaler.scale_, model.coef_.ravel()):
            original_beta = float(beta / scale)
            coefficients.append({
                "model": model_name,
                "term": feature,
                "selected": int(abs(beta) > COEF_TOL),
                "imputation_median": float(median),
                "standardization_mean_after_imputation": float(mean),
                "standardization_SD_after_imputation": float(scale),
                "standardized_coefficient": float(beta),
                "OR_per_1_SD": float(np.exp(beta)),
                "coefficient_per_original_unit": original_beta,
                "OR_per_original_unit": float(np.exp(original_beta)),
            })
    return pd.concat(grids, ignore_index=True), pd.DataFrame(choices), pd.DataFrame(coefficients)


def bootstrap_oof_metrics(mean_oof: pd.DataFrame, n_boot: int):
    rng = np.random.default_rng(SEED + 700000)
    rows = []
    for model, group in mean_oof.groupby("model"):
        y = group.observed.to_numpy()
        p = group.predicted_probability.to_numpy()
        event_idx = np.flatnonzero(y == 1)
        nonevent_idx = np.flatnonzero(y == 0)
        metric_names = ["AUC", "AUPRC", "Brier", "Log loss"]
        boot_values = {m: [] for m in metric_names}
        for _ in range(n_boot):
            idx = np.concatenate([
                rng.choice(event_idx, size=len(event_idx), replace=True),
                rng.choice(nonevent_idx, size=len(nonevent_idx), replace=True),
            ])
            yb, pb = y[idx], p[idx]
            values = {
                "AUC": roc_auc_score(yb, pb),
                "AUPRC": average_precision_score(yb, pb),
                "Brier": brier_score_loss(yb, pb),
                "Log loss": log_loss(yb, pb, labels=[0, 1]),
            }
            for metric, value in values.items():
                boot_values[metric].append(float(value))
        point = performance(y, p)
        for metric, values in boot_values.items():
            rows.append({
                "model": model,
                "metric": metric,
                "point_estimate_from_mean_repeated_oof_probability": point[metric],
                "bootstrap_95_CI_low": float(np.quantile(values, 0.025)),
                "bootstrap_95_CI_high": float(np.quantile(values, 0.975)),
                "bootstrap_resamples": n_boot,
                "note": "Patient-level stratified bootstrap of fixed mean repeated OOF predictions; pipeline was not refitted",
            })
    return pd.DataFrame(rows)


def calibration_groups_ridge(mean_oof: pd.DataFrame, groups: int = 8):
    d = mean_oof.loc[mean_oof.model == "ridge"].copy()
    d["risk_group"] = pd.qcut(d.predicted_probability, q=groups, labels=False, duplicates="drop") + 1
    rows = []
    for group, g in d.groupby("risk_group"):
        n, events = len(g), int(g.observed.sum())
        phat = events / n
        z = 1.96
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
        rows.append({
            "risk_group": int(group), "n": n, "events": events,
            "mean_predicted_probability": float(g.predicted_probability.mean()),
            "observed_event_proportion": phat,
            "observed_wilson_95_CI_low": center - half,
            "observed_wilson_95_CI_high": center + half,
        })
    return pd.DataFrame(rows)


def threshold_metrics_ridge(mean_oof: pd.DataFrame):
    d = mean_oof.loc[mean_oof.model == "ridge"]
    y, p = d.observed.to_numpy(), d.predicted_probability.to_numpy()
    rows = []
    for threshold in np.arange(0.05, 0.61, 0.05):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "accuracy": (tp + tn) / len(y),
        })
    return pd.DataFrame(rows)


def decision_curve_ridge(mean_oof: pd.DataFrame):
    d = mean_oof.loc[mean_oof.model == "ridge"]
    y, p, n = d.observed.to_numpy(), d.predicted_probability.to_numpy(), len(d)
    prevalence = y.mean()
    rows = []
    for threshold in np.linspace(0.05, 0.50, 91):
        pred = p >= threshold
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        weight = threshold / (1 - threshold)
        rows.append({
            "threshold": threshold,
            "ridge_model": tp / n - fp / n * weight,
            "treat_all": prevalence - (1 - prevalence) * weight,
            "treat_none": 0.0,
        })
    return pd.DataFrame(rows)


def save_figures(out: Path, mean_oof: pd.DataFrame, calibration: pd.DataFrame, dca: pd.DataFrame):
    colors = {"ridge": "#1f77b4", "elastic_net": "#2ca02c", "lasso": "#ff7f0e"}
    plt.figure(figsize=(7, 6))
    for model, g in mean_oof.groupby("model"):
        fpr, tpr, _ = roc_curve(g.observed, g.predicted_probability)
        auc = roc_auc_score(g.observed, g.predicted_probability)
        plt.plot(fpr, tpr, lw=2, color=colors[model], label=f"{model} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="0.55")
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.title("Mean repeated nested OOF ROC curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figure_roc_nested_cv.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 6))
    for model, g in mean_oof.groupby("model"):
        precision, recall, _ = precision_recall_curve(g.observed, g.predicted_probability)
        ap = average_precision_score(g.observed, g.predicted_probability)
        plt.plot(recall, precision, lw=2, color=colors[model], label=f"{model} (AUPRC={ap:.3f})")
    plt.axhline(mean_oof.observed.mean(), ls="--", color="0.55", label="Event rate")
    plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision (PPV)"); plt.title("Mean repeated nested OOF precision-recall curves"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figure_precision_recall_nested_cv.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 6))
    low = calibration.observed_event_proportion - calibration.observed_wilson_95_CI_low
    high = calibration.observed_wilson_95_CI_high - calibration.observed_event_proportion
    plt.errorbar(calibration.mean_predicted_probability, calibration.observed_event_proportion, yerr=np.vstack([low, high]), fmt="o-", capsize=3, label="Ridge")
    plt.plot([0, 1], [0, 1], "--", color="0.55", label="Ideal")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.xlabel("Predicted probability"); plt.ylabel("Observed event proportion"); plt.title("Ridge calibration (mean repeated nested OOF)"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figure_calibration_ridge.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(dca.threshold, dca.ridge_model, lw=2, label="Ridge model")
    plt.plot(dca.threshold, dca.treat_all, "--", label="Treat all")
    plt.plot(dca.threshold, dca.treat_none, "--", label="Treat none")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Exploratory decision-curve analysis"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "figure_decision_curve_ridge.png", dpi=300); plt.close()


def write_results_readme(out: Path, checks: dict, summary: pd.DataFrame):
    ridge = summary.loc[summary.model.eq("ridge")].set_index("metric")
    text = f"""# Primary analysis results

## Candidate predictors

Male sex; weight; preoperative CRP; preoperative neutrophils; driving pressure at OLV 30 min; PEEP at OLV 30 min; PaCO2 at OLV 30 min; blood base excess at OLV 30 min; heart rate at OLV 30 min; and Ti/Ttot at OLV 30 min. Age and intraoperative P/F ratio were not included.

Ti/Ttot and driving pressure were read directly from their existing source-workbook columns. Ti/Ttot was independently checked against inspiratory time (seconds) × respiratory rate (breaths/minute) / 60, and driving pressure was independently checked against Pplat - PEEP. The reconstructed values were not used to overwrite the recorded source values.

## Dataset checks

- Patients: {checks['n_patients']}
- Events: {checks['n_events']} ({checks['event_rate']:.3%})
- Maximum absolute discrepancy for driving pressure versus Pplat - PEEP: {checks['max_absolute_driving_pressure_formula_error']:.12g}
- Driving-pressure rows checked / mismatches: {checks['n_rows_driving_pressure_formula_checked']} / {checks['n_rows_driving_pressure_formula_mismatch']}
- Maximum absolute discrepancy for Ti/Ttot versus Ti × RR / 60: {checks['max_absolute_ti_ttot_formula_error']:.12g}
- Ti/Ttot rows checked / mismatches: {checks['n_rows_ti_ttot_formula_checked']} / {checks['n_rows_ti_ttot_formula_mismatch']}
## Validation design

The primary model was ridge logistic regression. Elastic-net and LASSO were secondary comparisons. The outer validation was stratified 5-fold cross-validation repeated 10 times. Within each outer training fold, a separate stratified 5-fold inner loop performed median imputation, standardisation and hyperparameter tuning by minimum log loss. The untouched outer fold was used only for performance estimation.

## Primary ridge results

- Mean repeated nested-CV AUC: {ridge.loc['AUC', 'mean_across_10_repeats']:.6f} (SD {ridge.loc['AUC', 'sd_across_10_repeats']:.6f})
- Mean repeated nested-CV AUPRC: {ridge.loc['AUPRC', 'mean_across_10_repeats']:.6f}
- Mean repeated nested-CV Brier score: {ridge.loc['Brier', 'mean_across_10_repeats']:.6f}
- Mean calibration intercept: {ridge.loc['Calibration intercept', 'mean_across_10_repeats']:.6f}
- Mean calibration slope: {ridge.loc['Calibration slope', 'mean_across_10_repeats']:.6f}

The resampling percentiles in the performance summary describe variability across the 10 repeated partitions; they are not an external-validation confidence interval. The bootstrap CI file resamples fixed mean repeated OOF predictions and does not repeat model development.

## Generated output files

- `candidate_variable_definitions_and_missingness.csv`: exact source columns, units, derivation and missingness.
- `candidate_descriptive_statistics.csv`: distributions overall and by outcome.
- `candidate_spearman_correlation.csv` and `candidate_vif.csv`: collinearity checks.
- `inner_tuning_results.csv`: every tested hyperparameter combination in every outer training set.
- `selected_hyperparameters_by_outer_fold.csv`: the selected setting for each outer fold.
- `performance_by_outer_fold.csv` and `performance_by_repeat.csv`: granular validation results.
- `model_performance_summary.csv`: principal summary; report the mean across repeats.
- `repeated_nested_oof_predictions.csv`: all 10 predictions per patient per model; generated locally but omitted from the public reference-results archive.
- `mean_repeated_oof_predictions.csv`: patient-level mean of the 10 OOF predictions; generated locally but omitted from the public reference-results archive.
- `selection_stability.csv`: descriptive elastic-net/LASSO selection frequencies.
- `final_model_coefficients_descriptive.csv`: post-development full-cohort fit for the potential equation; not an unbiased performance estimate and not for conventional p-values.
"""
    (out / "README_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    start = time.time()
    input_path = args.input.resolve()
    out = args.output.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")
    out.mkdir(parents=True, exist_ok=True)

    _, x, y, checks = load_data(input_path)
    print(f"Loaded {len(y)} patients, {int(y.sum())} events, {x.shape[1]} fixed predictors")
    definitions, descriptive = candidate_tables(x, y)
    definitions.to_csv(out / "candidate_variable_definitions_and_missingness.csv", index=False, encoding="utf-8-sig")
    descriptive.to_csv(out / "candidate_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    x.corr(method="spearman").to_csv(out / "candidate_spearman_correlation.csv", encoding="utf-8-sig")
    vif_table(x).to_csv(out / "candidate_vif.csv", index=False, encoding="utf-8-sig")

    predictions, selected_hyper, tuning, coefficients = run_nested_cv(x, y)
    summary, by_repeat, mean_oof = summarize_predictions(predictions)
    by_fold = fold_performance(predictions)
    stability = selection_stability(coefficients)
    full_grid, full_choices, full_coefficients = fit_full_cohort_models(x, y)
    bootstrap_ci = bootstrap_oof_metrics(mean_oof, args.bootstrap)
    calibration = calibration_groups_ridge(mean_oof)
    thresholds = threshold_metrics_ridge(mean_oof)
    dca = decision_curve_ridge(mean_oof)

    predictions.to_csv(out / "repeated_nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    mean_oof.to_csv(out / "mean_repeated_oof_predictions.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(out / "inner_tuning_results.csv", index=False, encoding="utf-8-sig")
    selected_hyper.to_csv(out / "selected_hyperparameters_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(out / "performance_by_outer_fold.csv", index=False, encoding="utf-8-sig")
    by_repeat.to_csv(out / "performance_by_repeat.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "model_performance_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(out / "selection_stability.csv", index=False, encoding="utf-8-sig")
    full_grid.to_csv(out / "final_full_cohort_tuning_grid.csv", index=False, encoding="utf-8-sig")
    full_choices.to_csv(out / "final_full_cohort_hyperparameters.csv", index=False, encoding="utf-8-sig")
    full_coefficients.to_csv(out / "final_model_coefficients_descriptive.csv", index=False, encoding="utf-8-sig")
    bootstrap_ci.to_csv(out / "performance_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(out / "calibration_groups_ridge.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(out / "threshold_metrics_ridge.csv", index=False, encoding="utf-8-sig")
    dca.to_csv(out / "decision_curve_values_ridge.csv", index=False, encoding="utf-8-sig")
    save_figures(out, mean_oof, calibration, dca)

    metadata = {
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": input_path.name,
        "input_sha256": file_sha256(input_path),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
        "random_seed": SEED,
        "outer_splits": OUTER_SPLITS,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits": INNER_SPLITS,
        "C_grid": C_GRID,
        "elastic_net_l1_ratio_grid": L1_GRID,
        "primary_model": "ridge logistic regression",
        "secondary_models": ["elastic-net logistic regression", "LASSO logistic regression"],
        "hyperparameter_selection_metric": "minimum inner cross-validated log loss",
        "missing_data_strategy": "median imputation within each resampling training split",
        "scaling_strategy": "standardisation within each resampling training split",
        "fixed_features": FEATURES,
        "source_checks": checks,
        "bootstrap_resamples_for_fixed_mean_oof_predictions": args.bootstrap,
        "elapsed_seconds": time.time() - start,
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_results_readme(out, checks, summary)
    print(f"Analysis completed in {(time.time() - start) / 60:.2f} minutes")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
