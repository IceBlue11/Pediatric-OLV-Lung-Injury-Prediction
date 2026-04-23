import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN_CANDIDATES = ("Outcome", "Label", "label")
BASELINE_VARIABLE_LABELS = {
    "Age (months)": "Age (months)",
    "Is_male": "Male sex",
    "Weight": "Weight (kg)",
    "Height": "Height (cm)",
    "If_left": "Left-sided procedure",
    "Preoperative X or CT": "Preoperative inflammatory imaging",
}


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


def summarize_continuous(series: pd.Series, method: str):
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return {
            "summary": "NA",
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }
    mean = clean_series.mean()
    sd = clean_series.std()
    median = clean_series.median()
    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    summary = f"{median:.2f} [{q1:.2f}, {q3:.2f}]" if method == "median_iqr" else f"{mean:.2f} +/- {sd:.2f}"
    return {
        "summary": summary,
        "n": int(clean_series.shape[0]),
        "mean": mean,
        "sd": sd,
        "median": median,
        "q1": q1,
        "q3": q3,
    }


def summarize_binary(series: pd.Series):
    clean_series = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    total = int(clean_series.shape[0])
    positive = int((clean_series == 1).sum())
    percent = (positive / total * 100.0) if total else 0.0
    return {
        "summary": f"{positive} ({percent:.1f}%)",
        "n": positive,
        "percent": percent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument("--data-path", default="data/raw/original_data_p6e.xlsx")
    parser.add_argument("--summary", choices=["mean_sd", "median_iqr"], default="mean_sd")
    args = parser.parse_args()

    extra_dir = Path("results") / f"run_{args.run_id}" / "extra_pack"
    extra_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.data_path))
    target_column = resolve_target_column(df)
    rows = []

    for variable, display_name in BASELINE_VARIABLE_LABELS.items():
        if variable not in df.columns:
            continue
        series = df[variable]
        binary_values = pd.to_numeric(series, errors="coerce").dropna()
        if not binary_values.empty and binary_values.isin([0, 1]).all():
            summary = summarize_binary(series)
            rows.append(
                {
                    "variable": variable,
                    "display_name": display_name,
                    "type": "binary",
                    "level": "1",
                    "summary": summary["summary"],
                    "n": summary["n"],
                    "percent": summary["percent"],
                    "mean": np.nan,
                    "sd": np.nan,
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                }
            )
        else:
            summary = summarize_continuous(series, method=args.summary)
            rows.append(
                {
                    "variable": variable,
                    "display_name": display_name,
                    "type": "continuous",
                    "level": "",
                    "summary": summary["summary"],
                    "n": summary["n"],
                    "percent": np.nan,
                    "mean": summary["mean"],
                    "sd": summary["sd"],
                    "median": summary["median"],
                    "q1": summary["q1"],
                    "q3": summary["q3"],
                }
            )

    pd.DataFrame(rows).to_csv(extra_dir / "baseline_table.csv", index=False)
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)
    event_rate = {
        "n": int(len(y)),
        "events": int((y == 1).sum()),
        "event_rate": float((y == 1).mean()),
    }
    with (extra_dir / "event_rate.json").open("w", encoding="utf-8") as handle:
        json.dump(event_rate, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
