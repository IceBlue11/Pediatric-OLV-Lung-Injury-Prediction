import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument(
        "--oof-path",
        default="results/run_20260319_183800/extra_pack/oof_predictions_RFECVLR.csv",
    )
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    oof_path = Path(args.oof_path)
    if not oof_path.exists():
        oof_path = Path("results") / f"run_{args.run_id}" / "extra_pack" / "oof_predictions_RFECVLR.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    df = pd.read_csv(oof_path)
    y_true = df["y_true"].astype(int).values
    y_prob = df["y_prob"].astype(float).values

    brier = brier_score_loss(y_true, y_prob)

    eps = 1e-6
    p_clip = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)

    calib_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    calib_model.fit(logit, y_true)
    intercept = float(calib_model.intercept_[0])
    slope = float(calib_model.coef_[0][0])

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=args.bins, strategy="uniform"
    )

    bins = np.linspace(0, 1, args.bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    bin_rows = []
    for i in range(1, args.bins + 1):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        bin_rows.append(
            {
                "bin": i,
                "bin_lower": bins[i - 1],
                "bin_upper": bins[i],
                "mean_pred": float(np.mean(y_prob[mask])),
                "frac_positive": float(np.mean(y_true[mask])),
                "count": int(np.sum(mask)),
            }
        )

    run_root = Path("results") / f"run_{args.run_id}" / "extra_pack"
    run_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(bin_rows).to_csv(run_root / "calibration_curve.csv", index=False)

    metrics = {
        "brier_score": brier,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "bins": args.bins,
    }
    with open(run_root / "calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_root / "calibration.svg", format="svg", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
