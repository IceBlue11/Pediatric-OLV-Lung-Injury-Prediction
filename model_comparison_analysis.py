import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")
MODEL_OUTPUT_NAME = {
    "RFECV+LR": "RFECVLR",
    "LR": "LR",
    "BP": "BP",
    "AdaBoost": "ADA",
    "RF": "RF",
    "DT": "DT",
}
MODEL_ORDER = ["RFECV+LR", "LR", "BP", "AdaBoost", "RF", "DT"]


def clean_name(name: str):
    if name is None:
        return name
    value = str(name).replace("\t", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_table(data_path: Path):
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path, sheet_name="Sheet1")
    df.columns = [clean_name(col) for col in df.columns]
    return df


def resolve_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    if len(df.columns) >= 69:
        return str(df.columns[-1])
    raise ValueError(f"Target column not found. Tried: {TARGET_COLUMN_CANDIDATES} and final-column fallback.")


def load_data(data_path: Path):
    df = read_table(data_path)
    target_column = resolve_target_column(df)
    feature_columns = df.columns[1:68]
    if len(feature_columns) != 67:
        raise ValueError(f"Expected 67 features, got {len(feature_columns)}.")
    x = df[feature_columns].copy()
    y = df[target_column].copy().fillna(0).astype(int)
    for col in x.columns:
        if x[col].dtype == "object":
            mask = x[col].astype(str).str.startswith("=", na=False)
            x.loc[mask, col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    categorical_candidates = ["Gender", "Primary Disease", "Surgical Procedure"]
    categorical_columns = [col for col in categorical_candidates if col in x.columns]
    x_processed = x.copy()
    numerical_columns = [col for col in x_processed.columns if col not in categorical_columns]
    for col in numerical_columns:
        x_processed[col] = x_processed[col].fillna(x_processed[col].median())
    for col in categorical_columns:
        mode_value = x_processed[col].mode()
        fill_value = mode_value.iloc[0] if not mode_value.empty else 0
        x_processed[col] = x_processed[col].fillna(fill_value)
    if "id" in df.columns:
        ids = df["id"].astype(str).copy()
    else:
        ids = pd.Series([f"S{i + 1:04d}" for i in range(len(df))])
    return x_processed, y, feature_columns, categorical_columns, ids


def load_selected_features(selected_features_path: Path, feature_columns):
    selected_df = pd.read_csv(selected_features_path)
    if "Feature" not in selected_df.columns:
        raise ValueError(f"'Feature' column not found in {selected_features_path}.")
    selected = selected_df["Feature"].astype(str).tolist()
    direct_map = {str(col): col for col in feature_columns}
    stripped_map = {str(col).replace("\t", " ").strip(): col for col in feature_columns}
    resolved = []
    unresolved = []
    for feature in selected:
        if feature in direct_map:
            resolved.append(direct_map[feature])
            continue
        key = feature.replace("\t", " ").strip()
        if key in stripped_map:
            resolved.append(stripped_map[key])
        else:
            unresolved.append(feature)
    if unresolved:
        raise ValueError(f"Selected features missing in data columns: {unresolved}")
    return resolved


def build_models(random_state: int):
    return {
        "RFECV+LR": LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced"),
        "LR": LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced"),
        "BP": MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100,)),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
        "RF": RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight="balanced_subsample"),
        "DT": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
    }


def prepare_fold_data(x_train: pd.DataFrame, x_test: pd.DataFrame, categorical_columns: list[str]):
    x_train_processed = x_train.copy()
    x_test_processed = x_test.copy()
    numerical_columns = [col for col in x_train_processed.columns if col not in categorical_columns]
    for col in numerical_columns:
        median_value = x_train_processed[col].median()
        x_train_processed[col] = x_train_processed[col].fillna(median_value)
        x_test_processed[col] = x_test_processed[col].fillna(median_value)
    for col in categorical_columns:
        if col in x_train_processed.columns:
            mode_value = x_train_processed[col].mode()
            fill_value = mode_value.iloc[0] if not mode_value.empty else 0
            x_train_processed[col] = x_train_processed[col].fillna(fill_value)
            x_test_processed[col] = x_test_processed[col].fillna(fill_value)
    return x_train_processed, x_test_processed


def run_comparison(
    x_full: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    selected_features: list[str],
    categorical_columns: list[str],
    cv_folds: int,
    random_state: int,
):
    models = build_models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows_by_model = {name: [] for name in MODEL_ORDER}
    auc_by_model = {name: [] for name in MODEL_ORDER}

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_full, y), 1):
        for model_name in MODEL_ORDER:
            model = models[model_name]
            if model_name == "RFECV+LR":
                x_train = x_full.iloc[train_idx][selected_features]
                x_test = x_full.iloc[test_idx][selected_features]
            else:
                x_train = x_full.iloc[train_idx]
                x_test = x_full.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            x_train_processed, x_test_processed = prepare_fold_data(
                x_train=x_train,
                x_test=x_test,
                categorical_columns=[col for col in categorical_columns if col in x_train.columns],
            )
            if model_name in ["RFECV+LR", "LR", "BP"]:
                scaler = StandardScaler()
                x_train_ready = scaler.fit_transform(x_train_processed)
                x_test_ready = scaler.transform(x_test_processed)
            else:
                x_train_ready = x_train_processed.values
                x_test_ready = x_test_processed.values
            if np.isnan(x_train_ready).any() or np.isnan(x_test_ready).any():
                x_train_ready = np.nan_to_num(x_train_ready)
                x_test_ready = np.nan_to_num(x_test_ready)
            model.fit(x_train_ready, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(x_test_ready)[:, 1]
            else:
                y_prob = model.predict(x_test_ready)
            auc_by_model[model_name].append(float(roc_auc_score(y_test, y_prob)))
            y_pred_05 = (y_prob >= 0.5).astype(int)
            y_test_values = y_test.values
            for i, data_idx in enumerate(test_idx.tolist()):
                rows_by_model[model_name].append(
                    {
                        "sample_index": int(data_idx),
                        "id": str(ids.iloc[data_idx]),
                        "fold": int(fold),
                        "y_true": int(y_test_values[i]),
                        "y_prob": float(y_prob[i]),
                        "y_pred_0.5": int(y_pred_05[i]),
                    }
                )

    oof_by_model = {}
    for model_name, rows in rows_by_model.items():
        oof_by_model[model_name] = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    return oof_by_model, auc_by_model


def pooled_metrics(df: pd.DataFrame):
    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    return {
        "Accuracy": float((tn + tp) / total),
        "F1": float(f1_score(y_true, y_pred)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_metrics_table(oof_by_model: dict, auc_by_model: dict):
    rows = []
    for model_name in MODEL_ORDER:
        pooled = pooled_metrics(oof_by_model[model_name])
        rows.append(
            {
                "Model": model_name,
                "Accuracy": pooled["Accuracy"],
                "F1": pooled["F1"],
                "AUC_mean": float(np.mean(auc_by_model[model_name])),
                "AUC_std": float(np.std(auc_by_model[model_name])),
                "Sensitivity": pooled["Sensitivity"],
                "Specificity": pooled["Specificity"],
            }
        )
    return pd.DataFrame(rows)


def build_confusion_table(oof_by_model: dict):
    rows = []
    for model_name in MODEL_ORDER:
        pooled = pooled_metrics(oof_by_model[model_name])
        rows.append(
            {
                "model": model_name,
                "tn": pooled["tn"],
                "fp": pooled["fp"],
                "fn": pooled["fn"],
                "tp": pooled["tp"],
            }
        )
    return pd.DataFrame(rows)


def build_auc_folds_table(auc_by_model: dict):
    rows = []
    for model_name in MODEL_ORDER:
        for fold, auc in enumerate(auc_by_model[model_name], 1):
            rows.append({"model": model_name, "fold": int(fold), "auc": float(auc)})
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    oof_by_model: dict,
    auc_by_model: dict,
    selected_features: list[str],
    run_id: str,
    data_path: Path,
    cv_folds: int,
    random_state: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = build_metrics_table(oof_by_model=oof_by_model, auc_by_model=auc_by_model)
    confusion_df = build_confusion_table(oof_by_model=oof_by_model)
    auc_folds_df = build_auc_folds_table(auc_by_model=auc_by_model)
    metrics_df.to_csv(output_dir / "metrics_table1.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrices.csv", index=False)
    auc_folds_df.to_csv(output_dir / "auc_folds.csv", index=False)
    for model_name in MODEL_ORDER:
        short_name = MODEL_OUTPUT_NAME[model_name]
        oof_by_model[model_name][["id", "fold", "y_true", "y_prob", "y_pred_0.5"]].to_csv(
            output_dir / f"oof_predictions_{short_name}.csv",
            index=False,
        )
    metadata = {
        "run_id": run_id,
        "data_path": str(data_path),
        "cv_folds": cv_folds,
        "random_state": random_state,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "models": MODEL_ORDER,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate six-model comparison artifacts for the pediatric OLV study.")
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument("--data-path", default="data/raw/original_data_p6e.xlsx")
    parser.add_argument("--selected-features-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    run_root = Path("results") / f"run_{args.run_id}"
    selected_features_path = (
        Path(args.selected_features_path)
        if args.selected_features_path
        else run_root / "legacy_pack" / "rfecv_selected_features.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "comparison_pack"
    x_full, y, feature_columns, categorical_columns, ids = load_data(Path(args.data_path))
    selected_features = load_selected_features(selected_features_path, feature_columns)
    oof_by_model, auc_by_model = run_comparison(
        x_full=x_full,
        y=y,
        ids=ids,
        selected_features=selected_features,
        categorical_columns=categorical_columns,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    write_outputs(
        output_dir=output_dir,
        oof_by_model=oof_by_model,
        auc_by_model=auc_by_model,
        selected_features=selected_features,
        run_id=args.run_id,
        data_path=Path(args.data_path),
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )
    print(output_dir.as_posix())


if __name__ == "__main__":
    main()
