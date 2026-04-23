import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def decision_curve(y_true, y_prob, thresholds):
    n = len(y_true)
    prevalence = np.mean(y_true)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        net_benefit = (tp / n) - (fp / n) * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        treat_none = 0.0

        rows.append(
            {
                "threshold": t,
                "net_benefit_model": net_benefit,
                "net_benefit_all": treat_all,
                "net_benefit_none": treat_none,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument(
        "--oof-path",
        default="results/run_20260319_183800/extra_pack/oof_predictions_RFECVLR.csv",
    )
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--t-max", type=float, default=0.60)
    parser.add_argument("--t-step", type=float, default=0.01)
    args = parser.parse_args()

    oof_path = Path(args.oof_path)
    if not oof_path.exists():
        oof_path = Path("results") / f"run_{args.run_id}" / "extra_pack" / "oof_predictions_RFECVLR.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    df = pd.read_csv(oof_path)
    y_true = df["y_true"].astype(int).values
    y_prob = df["y_prob"].astype(float).values

    thresholds = np.round(np.arange(args.t_min, args.t_max + 1e-9, args.t_step), 2)
    dca_df = decision_curve(y_true, y_prob, thresholds)

    run_root = Path("results") / f"run_{args.run_id}" / "extra_pack"
    run_root.mkdir(parents=True, exist_ok=True)

    dca_df.to_csv(run_root / "dca_values.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model", linewidth=2)
    plt.plot(dca_df["threshold"], dca_df["net_benefit_all"], label="Treat All", linestyle="--")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_none"], label="Treat None", linestyle="--")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_root / "dca.svg", format="svg", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
