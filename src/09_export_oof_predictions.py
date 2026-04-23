import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")


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


def load_data(data_path: Path, sheet_name: str = "Sheet1"):
    df = read_table(data_path=data_path, sheet_name=sheet_name)
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
    y = df[target_column].copy().fillna(0).astype(int)
    return df, x, y, feature_columns


def get_categorical_columns(df: pd.DataFrame):
    candidates = ["Gender", "Primary Disease", "Surgical Procedure"]
    return [col for col in candidates if col in df.columns]


def load_selected_features(selected_path: Path):
    selected_df = pd.read_csv(selected_path)
    if "Feature" not in selected_df.columns:
        raise ValueError("rfecv_selected_features.csv must contain a 'Feature' column.")
    return selected_df["Feature"].map(clean_name).tolist()


def build_model(name: str, random_state: int):
    if name in {"RFECV+LR", "LR"}:
        return LogisticRegression(random_state=random_state, max_iter=1000, class_weight="balanced")
    if name == "BP":
        return MLPClassifier(random_state=random_state, max_iter=1000, hidden_layer_sizes=(100,))
    if name == "RF":
        return RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight="balanced_subsample")
    if name == "DT":
        return DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    raise ValueError(f"Unknown model name: {name}")


def export_oof_predictions(
    model_name: str,
    x_full: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    selected_features: list[str],
    categorical_columns: list[str],
    output_path: Path,
    cv_folds: int = 5,
    random_state: int = 42,
):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_full, y), 1):
        if model_name == "RFECV+LR":
            x_train = x_full.iloc[train_idx][selected_features]
            x_test = x_full.iloc[test_idx][selected_features]
        else:
            x_train = x_full.iloc[train_idx]
            x_test = x_full.iloc[test_idx]

        y_train = y.iloc[train_idx]
        x_train_processed = x_train.copy()
        x_test_processed = x_test.copy()

        numeric_columns = [col for col in x_train_processed.columns if col not in categorical_columns]
        for col in numeric_columns:
            median_value = x_train_processed[col].median()
            x_train_processed[col] = x_train_processed[col].fillna(median_value)
            x_test_processed[col] = x_test_processed[col].fillna(median_value)

        for col in categorical_columns:
            if col in x_train_processed.columns:
                mode_value = x_train_processed[col].mode()
                fill_value = mode_value.iloc[0] if not mode_value.empty else 0
                x_train_processed[col] = x_train_processed[col].fillna(fill_value)
                x_test_processed[col] = x_test_processed[col].fillna(fill_value)

        if model_name in {"RFECV+LR", "LR", "BP"}:
            scaler = StandardScaler()
            x_train_ready = scaler.fit_transform(x_train_processed)
            x_test_ready = scaler.transform(x_test_processed)
        else:
            x_train_ready = x_train_processed.values
            x_test_ready = x_test_processed.values

        if np.isnan(x_train_ready).any() or np.isnan(x_test_ready).any():
            x_train_ready = np.nan_to_num(x_train_ready)
            x_test_ready = np.nan_to_num(x_test_ready)

        model = build_model(model_name, random_state=random_state)
        model.fit(x_train_ready, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test_ready)[:, 1]
        else:
            y_prob = model.predict(x_test_ready)

        oof_prob[test_idx] = y_prob
        oof_fold[test_idx] = fold

    if np.isnan(oof_prob).any():
        raise ValueError(f"OOF predictions contain NaN values for model {model_name}")

    out_df = pd.DataFrame(
        {
            "id": ids.astype(str).values,
            "fold": oof_fold,
            "y_true": y.values,
            "y_prob": oof_prob,
        }
    )
    out_df["y_pred_0.5"] = (out_df["y_prob"] >= 0.5).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument("--data-path", default="data/raw/original_data_p6e.xlsx")
    args = parser.parse_args()

    run_root = Path("results") / f"run_{args.run_id}"
    legacy_dir = run_root / "legacy_pack"
    extra_dir = run_root / "extra_pack"
    selected_path = legacy_dir / "rfecv_selected_features.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected features: {selected_path}")

    df, x_full, y, _ = load_data(Path(args.data_path))
    categorical_columns = get_categorical_columns(df)
    selected_features = load_selected_features(selected_path)
    missing = [feature for feature in selected_features if feature not in x_full.columns]
    if missing:
        raise ValueError(f"Selected features not in data columns: {missing}")

    ids = df["id"].astype(str) if "id" in df.columns else pd.Series([f"S{i + 1:04d}" for i in range(len(df))])
    model_files = {
        "RFECV+LR": extra_dir / "oof_predictions_RFECVLR.csv",
        "LR": extra_dir / "oof_predictions_LR.csv",
        "BP": extra_dir / "oof_predictions_BP.csv",
        "RF": extra_dir / "oof_predictions_RF.csv",
        "DT": extra_dir / "oof_predictions_DT.csv",
    }

    for model_name, output_path in model_files.items():
        export_oof_predictions(
            model_name=model_name,
            x_full=x_full,
            y=y,
            ids=ids,
            selected_features=selected_features,
            categorical_columns=categorical_columns,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
