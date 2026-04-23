import argparse
from pathlib import Path

import pandas as pd


def safe_read_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    args = parser.parse_args()

    run_root = Path("results") / f"run_{args.run_id}"
    legacy_dir = run_root / "legacy_pack"
    extra_dir = run_root / "extra_pack"
    extra_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metrics_table = safe_read_csv(legacy_dir / "metrics_table1.csv")
    if metrics_table is not None and not metrics_table.empty:
        for _, metric_row in metrics_table.iterrows():
            model_name = metric_row["Model"]
            rows.extend(
                [
                    {
                        "field_name": f"Table1.{model_name}.Accuracy",
                        "new_value": metric_row["Accuracy"],
                        "evidence_path": str(legacy_dir / "metrics_table1.csv"),
                    },
                    {
                        "field_name": f"Table1.{model_name}.F1",
                        "new_value": metric_row["F1"],
                        "evidence_path": str(legacy_dir / "metrics_table1.csv"),
                    },
                    {
                        "field_name": f"Table1.{model_name}.AUC",
                        "new_value": f"{metric_row['AUC_mean']:.4f}+/-{metric_row['AUC_std']:.4f}",
                        "evidence_path": str(legacy_dir / "metrics_table1.csv"),
                    },
                    {
                        "field_name": f"Table1.{model_name}.Sensitivity",
                        "new_value": metric_row["Sensitivity"],
                        "evidence_path": str(legacy_dir / "metrics_table1.csv"),
                    },
                    {
                        "field_name": f"Table1.{model_name}.Specificity",
                        "new_value": metric_row["Specificity"],
                        "evidence_path": str(legacy_dir / "metrics_table1.csv"),
                    },
                ]
            )

    auc_ci = safe_read_csv(extra_dir / "auc_ci.csv")
    if auc_ci is not None and not auc_ci.empty:
        auc_row = auc_ci[auc_ci["metric"] == "auc"].iloc[0]
        rows.append(
            {
                "field_name": "RFECV+LR.AUC_95CI",
                "new_value": f"{auc_row['ci_lower']:.4f} to {auc_row['ci_upper']:.4f}",
                "evidence_path": str(extra_dir / "auc_ci.csv"),
            }
        )

    calibration_metrics = extra_dir / "calibration_metrics.json"
    if calibration_metrics.exists():
        rows.append(
            {
                "field_name": "Calibration.Brier",
                "new_value": "see calibration_metrics.json",
                "evidence_path": str(calibration_metrics),
            }
        )

    event_rate = extra_dir / "event_rate.json"
    if event_rate.exists():
        rows.append(
            {
                "field_name": "Event.Rate",
                "new_value": "see event_rate.json",
                "evidence_path": str(event_rate),
            }
        )

    pd.DataFrame(rows).to_csv(extra_dir / "patch_values_filled.csv", index=False)


if __name__ == "__main__":
    main()
