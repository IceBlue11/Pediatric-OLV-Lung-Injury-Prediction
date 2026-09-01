"""Sensitivity analyses for the pediatric OLV ridge model."""

import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_primary_analysis import (
    C_GRID,
    FEATURES,
    SEED,
    file_sha256,
    load_data,
)

MAX_ITER = 10000


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "data" / "original_data_p6e.xlsx")
    parser.add_argument("--output", type=Path, default=here / "results" / "sensitivity")
    parser.add_argument("--imputations", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    return parser.parse_args()


def metric_values(y, p):
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Log loss": float(log_loss(y, p, labels=[0, 1])),
    }


def make_median_pipe(c, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=MAX_ITER, random_state=seed)),
    ])


def make_iterative_pipe(c, seed):
    return Pipeline([
        ("imputer", IterativeImputer(
            estimator=BayesianRidge(), max_iter=10, sample_posterior=True,
            random_state=seed, skip_complete=True,
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=MAX_ITER, random_state=seed)),
    ])


def tune_median(x, y, seed, groups=None):
    if groups is None:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = splitter.split(x, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = splitter.split(x, y, groups=groups)
    splits = list(splits)
    rows = []
    for c in C_GRID:
        losses = []
        for j, (tr, va) in enumerate(splits):
            pipe = make_median_pipe(c, seed + j)
            pipe.fit(x.iloc[tr], y.iloc[tr])
            losses.append(log_loss(y.iloc[va], pipe.predict_proba(x.iloc[va])[:, 1], labels=[0, 1]))
        rows.append({"C": c, "mean_log_loss": np.mean(losses), "se_log_loss": np.std(losses, ddof=1) / np.sqrt(len(losses))})
    table = pd.DataFrame(rows).sort_values(["mean_log_loss", "C"]).reset_index(drop=True)
    return float(table.iloc[0].C), table


def fivefold_sensitivity(x, y, mode, m=5):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred_rows, tuning_rows = [], []
    for fold, (tr, te) in enumerate(outer.split(x, y), start=1):
        xtr, ytr = x.iloc[tr], y.iloc[tr]
        inner = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold).split(xtr, ytr))
        grid = []
        for c in C_GRID:
            losses = []
            for inner_fold, (itr, iva) in enumerate(inner, start=1):
                if mode == "iterative_multiple":
                    probs = []
                    for imp in range(m):
                        pipe = make_iterative_pipe(c, SEED + fold * 10000 + inner_fold * 100 + imp)
                        pipe.fit(xtr.iloc[itr], ytr.iloc[itr])
                        probs.append(pipe.predict_proba(xtr.iloc[iva])[:, 1])
                    probability = np.mean(probs, axis=0)
                else:
                    pipe = make_median_pipe(c, SEED + fold * 100 + inner_fold)
                    pipe.fit(xtr.iloc[itr], ytr.iloc[itr])
                    probability = pipe.predict_proba(xtr.iloc[iva])[:, 1]
                losses.append(log_loss(ytr.iloc[iva], probability, labels=[0, 1]))
            grid.append({"analysis": mode, "outer_fold": fold, "C": c, "mean_inner_log_loss": np.mean(losses)})
        grid = pd.DataFrame(grid).sort_values(["mean_inner_log_loss", "C"])
        chosen_c = float(grid.iloc[0].C)
        tuning_rows.append(grid)
        if mode == "iterative_multiple":
            probabilities = []
            for imp in range(m):
                pipe = make_iterative_pipe(chosen_c, SEED + fold * 10000 + 9000 + imp)
                pipe.fit(xtr, ytr)
                probabilities.append(pipe.predict_proba(x.iloc[te])[:, 1])
            probability = np.mean(probabilities, axis=0)
        else:
            pipe = make_median_pipe(chosen_c, SEED + fold)
            pipe.fit(xtr, ytr)
            probability = pipe.predict_proba(x.iloc[te])[:, 1]
        for idx, p in zip(te, probability):
            pred_rows.append({"analysis": mode, "outer_fold": fold, "row_index_zero_based": int(idx), "observed": int(y.iloc[idx]), "predicted_probability": float(p), "chosen_C": chosen_c})
        print(f"{mode}: outer fold {fold}/5 complete")
    predictions = pd.DataFrame(pred_rows)
    return predictions, pd.concat(tuning_rows, ignore_index=True), metric_values(predictions.observed, predictions.predicted_probability)


def bootstrap_optimism(x, y, n_boot=200):
    full_c, full_grid = tune_median(x, y, SEED + 800000)
    full_pipe = make_median_pipe(full_c, SEED + 800000)
    full_pipe.fit(x, y)
    apparent = metric_values(y, full_pipe.predict_proba(x)[:, 1])

    rng = np.random.default_rng(SEED + 810000)
    rows, coef_rows = [], []
    completed, attempts = 0, 0
    while completed < n_boot and attempts < n_boot * 5:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        xb = x.iloc[idx].reset_index(drop=True)
        yb = y.iloc[idx].reset_index(drop=True)
        if yb.nunique() < 2:
            continue
        groups = pd.Series(idx)
        try:
            c, _ = tune_median(xb, yb, SEED + 820000 + attempts, groups=groups)
            pipe = make_median_pipe(c, SEED + 830000 + attempts)
            pipe.fit(xb, yb)
            app = metric_values(yb, pipe.predict_proba(xb)[:, 1])
            test = metric_values(y, pipe.predict_proba(x)[:, 1])
            row = {"bootstrap": completed + 1, "C": c}
            for metric in apparent:
                row[f"apparent_{metric}"] = app[metric]
                row[f"test_{metric}"] = test[metric]
                row[f"optimism_{metric}"] = app[metric] - test[metric]
            rows.append(row)
            for feature, beta in zip(FEATURES, pipe.named_steps["model"].coef_.ravel()):
                coef_rows.append({"bootstrap": completed + 1, "feature": feature, "standardized_coefficient": float(beta)})
            completed += 1
            if completed % 20 == 0:
                print(f"Bootstrap optimism {completed}/{n_boot}")
        except ValueError:
            continue
    boot = pd.DataFrame(rows)
    summary = []
    for metric, app_value in apparent.items():
        optimism = boot[f"optimism_{metric}"]
        summary.append({
            "metric": metric,
            "full_cohort_apparent": app_value,
            "mean_optimism": float(optimism.mean()),
            "optimism_corrected": float(app_value - optimism.mean()),
            "optimism_p2_5": float(optimism.quantile(0.025)),
            "optimism_p97_5": float(optimism.quantile(0.975)),
            "bootstrap_resamples": n_boot,
        })
    coef = pd.DataFrame(coef_rows)
    coef_summary = coef.groupby("feature", as_index=False).agg(
        median_standardized_coefficient=("standardized_coefficient", "median"),
        coefficient_p2_5=("standardized_coefficient", lambda s: np.quantile(s, 0.025)),
        coefficient_p97_5=("standardized_coefficient", lambda s: np.quantile(s, 0.975)),
        positive_frequency=("standardized_coefficient", lambda s: np.mean(s > 0)),
    )
    return full_grid, boot, pd.DataFrame(summary), coef_summary


def main():
    args = parse_args()
    start = time.time()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")
    if args.imputations < 1:
        raise ValueError("--imputations must be at least 1")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    _, x, y, _ = load_data(input_path)
    iterative_pred, iterative_tuning, iterative_perf = fivefold_sensitivity(
        x, y, "iterative_multiple", m=args.imputations
    )

    complete = x.notna().all(axis=1)
    xcc, ycc = x.loc[complete].reset_index(drop=True), y.loc[complete].reset_index(drop=True)
    cc_pred, cc_tuning, cc_perf = fivefold_sensitivity(xcc, ycc, "complete_case", m=1)

    full_grid, boot, boot_summary, coef_summary = bootstrap_optimism(
        x, y, n_boot=args.bootstrap
    )

    iterative_pred.to_csv(output_dir / "sensitivity_iterative_multiple_oof_predictions.csv", index=False, encoding="utf-8-sig")
    iterative_tuning.to_csv(output_dir / "sensitivity_iterative_multiple_tuning.csv", index=False, encoding="utf-8-sig")
    cc_pred.to_csv(output_dir / "sensitivity_complete_case_oof_predictions.csv", index=False, encoding="utf-8-sig")
    cc_tuning.to_csv(output_dir / "sensitivity_complete_case_tuning.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"analysis": "Five posterior-sampling iterative imputations within each resampling split", "n": len(y), "events": int(y.sum()), **iterative_perf},
        {"analysis": "Complete-case nested five-fold CV", "n": len(ycc), "events": int(ycc.sum()), **cc_perf},
    ]).to_csv(output_dir / "missing_data_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    full_grid.to_csv(output_dir / "bootstrap_full_cohort_tuning_grid.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(output_dir / "bootstrap_full_pipeline_resamples.csv", index=False, encoding="utf-8-sig")
    boot_summary.to_csv(output_dir / "bootstrap_optimism_summary.csv", index=False, encoding="utf-8-sig")
    coef_summary.to_csv(output_dir / "bootstrap_coefficient_stability.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "input_file": input_path.name,
        "input_sha256": file_sha256(input_path),
        "iterative_imputations_per_split": args.imputations,
        "iterative_outer_folds": 5,
        "complete_case_n": int(len(ycc)),
        "complete_case_events": int(ycc.sum()),
        "bootstrap_resamples": args.bootstrap,
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "sensitivity_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"iterative": iterative_perf, "complete_case": cc_perf, "bootstrap": boot_summary.to_dict(orient="records"), **metadata}, indent=2))


if __name__ == "__main__":
    main()
