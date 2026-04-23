import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score


def compute_threshold_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def bootstrap_metrics(y_true, y_prob, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    auc_samples = []
    sens_samples = []
    spec_samples = []
    ppv_samples = []
    npv_samples = []
    f1_samples = []

    while len(auc_samples) < n_bootstrap:
        idx = rng.integers(0, n, n)
        y_true_b = y_true[idx]
        y_prob_b = y_prob[idx]

        if len(np.unique(y_true_b)) < 2:
            continue

        auc = roc_auc_score(y_true_b, y_prob_b)
        metrics = compute_threshold_metrics(y_true_b, y_prob_b, threshold=0.5)

        auc_samples.append(auc)
        sens_samples.append(metrics["sensitivity"])
        spec_samples.append(metrics["specificity"])
        ppv_samples.append(metrics["ppv"])
        npv_samples.append(metrics["npv"])
        f1_samples.append(metrics["f1"])

    return {
        "auc": np.array(auc_samples),
        "sensitivity": np.array(sens_samples),
        "specificity": np.array(spec_samples),
        "ppv": np.array(ppv_samples),
        "npv": np.array(npv_samples),
        "f1": np.array(f1_samples),
    }


def percentile_ci(values, alpha=0.95):
    lower = (1 - alpha) / 2 * 100
    upper = (1 + alpha) / 2 * 100
    return np.percentile(values, lower), np.percentile(values, upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument(
        "--oof-path",
        default="results/run_20260319_183800/extra_pack/oof_predictions_RFECVLR.csv",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    oof_path = Path(args.oof_path)
    if not oof_path.exists():
        oof_path = Path("results") / f"run_{args.run_id}" / "extra_pack" / "oof_predictions_RFECVLR.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    df = pd.read_csv(oof_path)
    y_true = df["y_true"].astype(int).values
    y_prob = df["y_prob"].astype(float).values

    point_auc = roc_auc_score(y_true, y_prob)
    point_metrics = compute_threshold_metrics(y_true, y_prob, threshold=0.5)

    samples = bootstrap_metrics(
        y_true, y_prob, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    rows = []
    for metric_name, point_value in [
        ("auc", point_auc),
        ("sensitivity", point_metrics["sensitivity"]),
        ("specificity", point_metrics["specificity"]),
        ("ppv", point_metrics["ppv"]),
        ("npv", point_metrics["npv"]),
        ("f1", point_metrics["f1"]),
    ]:
        ci_lower, ci_upper = percentile_ci(samples[metric_name])
        rows.append(
            {
                "metric": metric_name,
                "point_estimate": point_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_bootstrap": args.n_bootstrap,
                "seed": args.seed,
            }
        )

    run_root = Path("results") / f"run_{args.run_id}" / "extra_pack"
    run_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(run_root / "auc_ci.csv", index=False)
    pd.DataFrame({"auc": samples["auc"]}).to_csv(
        run_root / "bootstrap_samples_auc.csv", index=False
    )


if __name__ == "__main__":
    main()
