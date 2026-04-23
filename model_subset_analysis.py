import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")
ROUTINE_6 = [
    "Is_male",
    "Heart Rate (OLV-30min)",
    "Hemoglobin (OLV-30min)",
    "PaCO2 (OLV-30min)",
    "Base Excess of Blood (OLV-30min)",
    "Oxygenation Index (OLV-30min)",
]
CLINICAL_7 = [
    "Preoperative Neutrophils",
    "Heart Rate (OLV-30min)",
    "PaCO2 (OLV-30min)",
    "Base Excess of Blood (OLV-30min)",
    "Oxygenation Index (OLV-30min)",
    "PIP (OLV-30min)",
    "RR (Pre-OLV)",
]


def clean_name(name: str):
    if name is None:
        return name
    value = str(name).replace("	", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace("pseudo", "alpha")


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


def load_dataset(data_path: Path):
    df = read_table(data_path)
    target_column = resolve_target_column(df)
    feature_columns = [clean_name(col) for col in df.columns[1:68]]
    x = df[feature_columns].copy()
    for col in x.columns:
        if x[col].dtype == "object":
            formula_mask = x[col].astype(str).str.startswith("=", na=False)
            x.loc[formula_mask, col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    return x, y


def load_selected_features(selected_features_path: Path):
    selected_df = pd.read_csv(selected_features_path)
    if "Feature" not in selected_df.columns:
        raise ValueError(f"'Feature' column not found in {selected_features_path}.")
    return selected_df["Feature"].map(clean_name).tolist()


def evaluate_feature_set(x: pd.DataFrame, y: pd.Series, features: list[str], cv_folds: int = 5, random_state: int = 42):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    auc_values = []
    for train_idx, test_idx in cv.split(x, y):
        x_train = x.iloc[train_idx][features].copy()
        x_test = x.iloc[test_idx][features].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        for col in features:
            median_value = x_train[col].median()
            x_train[col] = x_train[col].fillna(median_value)
            x_test[col] = x_test[col].fillna(median_value)
        scaler = StandardScaler()
        x_train_ready = scaler.fit_transform(x_train)
        x_test_ready = scaler.transform(x_test)
        model = LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced")
        model.fit(x_train_ready, y_train)
        y_prob = model.predict_proba(x_test_ready)[:, 1]
        oof_prob[test_idx] = y_prob
        auc_values.append(float(roc_auc_score(y_test, y_prob)))
    y_true = y.to_numpy(dtype=int)
    y_pred = (oof_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "features": features,
        "n_features": len(features),
        "AUC_mean": float(np.mean(auc_values)),
        "AUC_std": float(np.std(auc_values)),
        "Accuracy": float((tn + tp) / (tn + fp + fn + tp)),
        "F1": float(f1_score(y_true, y_pred)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "AUC_pooled": float(roc_auc_score(y_true, oof_prob)),
        "Brier": float(np.mean((y_true - oof_prob) ** 2)),
        "oof_probabilities": oof_prob.tolist(),
    }


def forward_select(x: pd.DataFrame, y: pd.Series, feature_pool: list[str], target_k: int):
    selected = []
    history = []
    while len(selected) < target_k:
        best_candidate = None
        best_metrics = None
        best_key = None
        for feature in feature_pool:
            if feature in selected:
                continue
            trial = selected + [feature]
            metrics = evaluate_feature_set(x=x, y=y, features=trial)
            key = (metrics["AUC_mean"], metrics["F1"], metrics["Accuracy"])
            if best_key is None or key > best_key:
                best_candidate = feature
                best_metrics = metrics
                best_key = key
        selected.append(best_candidate)
        history.append(best_metrics)
    return history


def main():
    parser = argparse.ArgumentParser(description="Formal rerun for compact 5-8 predictor logistic models.")
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument("--data-path", default="data/raw/original_data_p6e.xlsx")
    args = parser.parse_args()

    run_root = Path("results") / f"run_{args.run_id}"
    legacy_dir = run_root / "legacy_pack"
    compact_dir = run_root / "compact_pack"
    compact_dir.mkdir(parents=True, exist_ok=True)

    selected_features_path = legacy_dir / "rfecv_selected_features.csv"
    if not selected_features_path.exists():
        raise FileNotFoundError(f"Missing selected features: {selected_features_path}")

    x, y = load_dataset(Path(args.data_path))
    selected_features = [feature for feature in load_selected_features(selected_features_path) if feature in x.columns]
    histories = {k: forward_select(x=x, y=y, feature_pool=selected_features, target_k=k) for k in range(5, 9)}

    candidate_sets = {
        "best_5": histories[5][-1]["features"],
        "best_6": histories[6][-1]["features"],
        "best_7": histories[7][-1]["features"],
        "best_8": histories[8][-1]["features"],
        "routine_6": [feature for feature in ROUTINE_6 if feature in x.columns],
        "clinical_7": [feature for feature in CLINICAL_7 if feature in x.columns],
    }

    metrics_rows = []
    feature_rows = []
    oof_rows = []
    for model_name, features in candidate_sets.items():
        metrics = evaluate_feature_set(x=x, y=y, features=features)
        metrics_rows.append(
            {
                "Model": model_name,
                "Predictors": metrics["n_features"],
                "AUC_mean": f"{metrics['AUC_mean']:.4f}",
                "AUC_std": f"{metrics['AUC_std']:.4f}",
                "Accuracy": f"{metrics['Accuracy']:.4f}",
                "F1": f"{metrics['F1']:.4f}",
                "Sensitivity": f"{metrics['Sensitivity']:.4f}",
                "Specificity": f"{metrics['Specificity']:.4f}",
                "AUC_pooled": f"{metrics['AUC_pooled']:.4f}",
                "AUC_ci_lower": "",
                "AUC_ci_upper": "",
                "Brier": f"{metrics['Brier']:.4f}",
                "Calibration_intercept": "",
                "Calibration_slope": "",
                "TN": metrics['TN'],
                "FP": metrics['FP'],
                "FN": metrics['FN'],
                "TP": metrics['TP'],
            }
        )
        feature_rows.append(
            {
                "Model": model_name,
                "Predictors": metrics["n_features"],
                "Features": "; ".join(features),
            }
        )
        for row_index, probability in enumerate(metrics["oof_probabilities"], 1):
            oof_rows.append(
                {
                    "Model": model_name,
                    "Row": row_index,
                    "Outcome": int(y.iloc[row_index - 1]),
                    "Probability": f"{probability:.8f}",
                }
            )

    screening_rows = []
    for k in range(5, 9):
        metrics = histories[k][-1]
        screening_rows.append(
            {
                "Predictors": k,
                "AUC_mean": f"{metrics['AUC_mean']:.4f}",
                "AUC_std": f"{metrics['AUC_std']:.4f}",
                "Accuracy": f"{metrics['Accuracy']:.4f}",
                "F1": f"{metrics['F1']:.4f}",
                "Features": "; ".join(metrics["features"]),
            }
        )

    pd.DataFrame(metrics_rows).to_csv(compact_dir / "compact_model_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(compact_dir / "compact_model_features.csv", index=False)
    pd.DataFrame(screening_rows).to_csv(compact_dir / "compact_model_screening_by_size.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(compact_dir / "compact_model_oof_predictions.csv", index=False)
    (compact_dir / "compact_model_feature_sets.json").write_text(json.dumps(candidate_sets, indent=2), encoding="utf-8")
    print(f"Wrote compact rerun artifacts to {compact_dir}")


if __name__ == "__main__":
    main()
