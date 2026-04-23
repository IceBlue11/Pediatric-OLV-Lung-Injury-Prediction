import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


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


def load_selected_features(selected_path: Path):
    selected_df = pd.read_csv(selected_path)
    if "Feature" not in selected_df.columns:
        raise ValueError("rfecv_selected_features.csv must contain a 'Feature' column.")
    return selected_df["Feature"].map(clean_name).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="20260319_183800")
    parser.add_argument("--data-path", default="data/raw/original_data_p6e.xlsx")
    args = parser.parse_args()

    run_root = Path("results") / f"run_{args.run_id}"
    legacy_dir = run_root / "legacy_pack"
    extra_dir = run_root / "extra_pack"
    extra_dir.mkdir(parents=True, exist_ok=True)

    selected_path = legacy_dir / "rfecv_selected_features.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected features: {selected_path}")

    df = read_table(Path(args.data_path))
    target_column = resolve_target_column(df)
    selected_features = load_selected_features(selected_path)
    missing = [feature for feature in selected_features if feature not in df.columns]
    if missing:
        raise ValueError(f"Selected features not in data columns: {missing}")

    x = df[selected_features].copy()
    y = pd.to_numeric(df[target_column], errors="coerce").fillna(0).astype(int)

    for col in x.columns:
        if x[col].dtype == "object":
            formula_mask = x[col].astype(str).str.startswith("=", na=False)
            x.loc[formula_mask, col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(x[col].median())

    scaler = StandardScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=selected_features)
    x_const = sm.add_constant(x_scaled, has_constant="add")

    try:
        model = sm.Logit(y, x_const).fit(disp=False, maxiter=200)
    except Exception:
        model = sm.GLM(y, x_const, family=sm.families.Binomial()).fit()

    params = model.params.drop("const")
    standard_errors = model.bse.drop("const")
    p_values = model.pvalues.drop("const")
    odds_ratios = np.exp(params)
    ci_lower = np.exp(params - 1.96 * standard_errors)
    ci_upper = np.exp(params + 1.96 * standard_errors)

    out_df = pd.DataFrame(
        {
            "Feature": params.index,
            "Beta": params.values,
            "SE": standard_errors.values,
            "OR": odds_ratios.values,
            "OR_95CI_lower": ci_lower.values,
            "OR_95CI_upper": ci_upper.values,
            "P_value": p_values.values,
        }
    )
    out_df.to_csv(extra_dir / "final_coef_or_table.csv", index=False)

    plot_df = out_df.sort_values("OR", ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(plot_df))
    plt.figure(figsize=(8, max(4, len(plot_df) * 0.35)))
    plt.errorbar(
        plot_df["OR"],
        y_positions,
        xerr=[plot_df["OR"] - plot_df["OR_95CI_lower"], plot_df["OR_95CI_upper"] - plot_df["OR"]],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=3,
    )
    plt.axvline(1, color="red", linestyle="--", linewidth=1)
    plt.yticks(y_positions, plot_df["Feature"])
    plt.xscale("log")
    plt.xlabel("Odds Ratio (log scale)")
    plt.title("Logistic Regression OR Forest Plot")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(extra_dir / "forest_or.svg", format="svg", dpi=600)
    plt.close()


if __name__ == "__main__":
    main()
