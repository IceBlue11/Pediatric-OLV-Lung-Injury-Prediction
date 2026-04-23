import datetime
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from train_rfecv_lr_model import (
    build_and_evaluate_logistic_model,
    custom_recursive_feature_elimination_lr,
    perform_statistical_tests,
    run_comparison_experiment,
)


TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")
DEPENDENCY_NAMES = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "scipy"]


def clean_name(name: str):
    if name is None:
        return name
    value = str(name).replace("	", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_table(data_path: Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
    df.columns = [clean_name(col) for col in df.columns]
    return df


def resolve_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    if len(df.columns) >= 69:
        return str(df.columns[-1])
    raise ValueError(f"Target column not found. Tried: {TARGET_COLUMN_CANDIDATES} and final-column fallback.")


def get_dependency_versions() -> dict:
    versions = {}
    for name in DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_metrics_table(comparison_results: dict) -> pd.DataFrame:
    model_order = ["RFECV+LR", "LR", "BP", "RF", "DT"]
    rows = []
    for model_name in model_order:
        rows.append(
            {
                "Model": model_name,
                "Accuracy": float(np.mean(comparison_results["accuracy"][model_name])),
                "F1": float(np.mean(comparison_results["f1"][model_name])),
                "AUC_mean": float(np.mean(comparison_results["auc"][model_name])),
                "AUC_std": float(np.std(comparison_results["auc"][model_name])),
                "Sensitivity": float(np.mean(comparison_results["sensitivity"][model_name])),
                "Specificity": float(np.mean(comparison_results["specificity"][model_name])),
            }
        )
    return pd.DataFrame(rows)


def build_patch_rows(data_summary: dict, metrics_df: pd.DataFrame, auc_values: list[float], evidence_root: Path):
    rows = [
        {
            "pdf_field": "Abstract.n",
            "old_value": "",
            "new_value": data_summary["total_samples"],
            "evidence_path": str(evidence_root / "data_summary.json"),
        },
        {
            "pdf_field": "Abstract.class_0",
            "old_value": "",
            "new_value": data_summary["class_0"],
            "evidence_path": str(evidence_root / "data_summary.json"),
        },
        {
            "pdf_field": "Abstract.class_1",
            "old_value": "",
            "new_value": data_summary["class_1"],
            "evidence_path": str(evidence_root / "data_summary.json"),
        },
    ]
    if auc_values:
        auc_mean = float(np.mean(auc_values))
        auc_std = float(np.std(auc_values))
        rows.append(
            {
                "pdf_field": "Abstract.AUC",
                "old_value": "",
                "new_value": f"{auc_mean:.3f}+/-{auc_std:.3f}",
                "evidence_path": str(evidence_root / "auc_folds.csv"),
            }
        )
    for _, metric_row in metrics_df.iterrows():
        rows.extend(
            [
                {
                    "pdf_field": f"Table1.{metric_row['Model']}.Accuracy",
                    "old_value": "",
                    "new_value": f"{metric_row['Accuracy']:.4f}",
                    "evidence_path": str(evidence_root / "metrics_table1.csv"),
                },
                {
                    "pdf_field": f"Table1.{metric_row['Model']}.F1",
                    "old_value": "",
                    "new_value": f"{metric_row['F1']:.4f}",
                    "evidence_path": str(evidence_root / "metrics_table1.csv"),
                },
                {
                    "pdf_field": f"Table1.{metric_row['Model']}.AUC",
                    "old_value": "",
                    "new_value": f"{metric_row['AUC_mean']:.4f}+/-{metric_row['AUC_std']:.4f}",
                    "evidence_path": str(evidence_root / "metrics_table1.csv"),
                },
                {
                    "pdf_field": f"Table1.{metric_row['Model']}.Sensitivity",
                    "old_value": "",
                    "new_value": f"{metric_row['Sensitivity']:.4f}",
                    "evidence_path": str(evidence_root / "metrics_table1.csv"),
                },
                {
                    "pdf_field": f"Table1.{metric_row['Model']}.Specificity",
                    "old_value": "",
                    "new_value": f"{metric_row['Specificity']:.4f}",
                    "evidence_path": str(evidence_root / "metrics_table1.csv"),
                },
            ]
        )
    return pd.DataFrame(rows)


def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path("results") / f"run_{run_id}" / "legacy_pack"
    run_root.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/raw/original_data_p6e.xlsx")
    if not data_path.exists():
        raise FileNotFoundError("No input dataset found. Expected data/raw/original_data_p6e.xlsx")

    df = read_table(data_path)
    target_column = resolve_target_column(df)
    feature_columns = [clean_name(col) for col in df.columns[1:68]]
    if len(feature_columns) != 67:
        raise ValueError(f"Expected 67 feature columns, got {len(feature_columns)}")

    x = df[feature_columns].copy()
    for col in x.columns:
        if x[col].dtype == "object":
            formula_mask = x[col].astype(str).str.startswith("=", na=False)
            x.loc[formula_mask, col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    categorical_columns = [col for col in ["Gender", "Primary Disease", "Surgical Procedure"] if col in x.columns]

    feature_results, selected_features, custom_rfecv = custom_recursive_feature_elimination_lr(
        x,
        y,
        categorical_columns,
        cv_folds=5,
        scoring="accuracy",
    )
    evaluation_results, final_model = build_and_evaluate_logistic_model(
        x,
        y,
        selected_features,
        categorical_columns,
        cv_folds=5,
    )

    x_processed = x.copy()
    for col in x_processed.columns:
        x_processed[col] = x_processed[col].fillna(x_processed[col].median())
    comparison_results, _ = run_comparison_experiment(
        x_processed,
        y,
        selected_features,
        categorical_columns,
        cv_folds=5,
    )
    _ = perform_statistical_tests(comparison_results, baseline_model="RFECV+LR")

    selected_df = feature_results[feature_results["Selected"]].copy()[["Feature", "Ranking"]].reset_index(drop=True)
    selected_df.to_csv(run_root / "rfecv_selected_features.csv", index=False)
    pd.DataFrame(
        {
            "n_features": custom_rfecv.n_features_history_,
            "cv_score": custom_rfecv.cv_scores_,
        }
    ).to_csv(run_root / "rfecv_curve_data.csv", index=False)

    auc_folds = pd.DataFrame(
        {
            "fold": list(range(1, len(evaluation_results["fold_aucs"]) + 1)),
            "auc_strict": evaluation_results["fold_aucs"],
            "auc_legacy": evaluation_results["fold_aucs"],
        }
    )
    auc_folds.to_csv(run_root / "auc_folds.csv", index=False)

    pooled_cm = confusion_matrix(
        np.asarray(evaluation_results["y_true"], dtype=int),
        (np.asarray(evaluation_results["y_pred_proba"], dtype=float) >= 0.5).astype(int),
    )
    pd.DataFrame(pooled_cm, columns=["pred_0", "pred_1"]).to_csv(run_root / "confusion_matrix.csv", index=False)

    metrics_df = build_metrics_table(comparison_results)
    metrics_df.to_csv(run_root / "metrics_table1.csv", index=False)
    (run_root / "metrics_table1.json").write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")

    data_summary = {
        "total_samples": int(len(df)),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
        "class_ratio": f"{int((y == 0).sum())}:{int((y == 1).sum())}",
        "original_features": int(len(feature_columns)),
        "selected_features": int(len(selected_features)),
    }
    (run_root / "data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")

    patch_df = build_patch_rows(data_summary, metrics_df, evaluation_results["fold_aucs"], run_root)
    patch_df.to_csv(run_root / "patch_values.csv", index=False)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": 42,
        "cv_folds": 5,
        "command": "src/legacy_rerun.py",
        "selected_feature_names": selected_features,
        "dependencies": get_dependency_versions(),
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return run_id, selected_features, evaluation_results, final_model


if __name__ == "__main__":
    main()
