import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, check=True)
    return result.returncode
def list_run_dirs(results_dir: Path):
    return {p.name for p in results_dir.glob("run_*") if p.is_dir()}
def detect_new_run(before, after):
    diff = sorted(after - before)
    if diff:
        return diff[-1]
    runs = sorted(after)
    return runs[-1] if runs else None
def clean_name(name: str) -> str:
    s = str(name)
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s
def make_paper_pack(run_id: str):
    run_root = Path("results") / f"run_{run_id}"
    legacy_dir = run_root / "legacy_pack"
    extra_dir = run_root / "extra_pack"
    pack_dir = run_root / "paper_pack_v1"
    pack_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = {
        extra_dir / "baseline_table.csv": pack_dir / "baseline_table.csv",
        legacy_dir / "metrics_table1.csv": pack_dir / "metrics_table1.csv",
        extra_dir / "final_coef_or_table.csv": pack_dir / "final_coef_or_table.csv",
        legacy_dir / "run_metadata.json": pack_dir / "run_metadata.json",
    }
    for src, dst in files_to_copy.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        dst.write_bytes(src.read_bytes())
    with open(legacy_dir / "data_summary.json", "r", encoding="utf-8") as f:
        data_summary = json.load(f)
    with open(extra_dir / "event_rate.json", "r", encoding="utf-8") as f:
        event_rate = json.load(f)
    metrics_table = pd.read_csv(legacy_dir / "metrics_table1.csv")
    auc_ci = pd.read_csv(extra_dir / "auc_ci.csv")
    calib_metrics = json.loads((extra_dir / "calibration_metrics.json").read_text(encoding="utf-8"))
    rfecv_features = pd.read_csv(legacy_dir / "rfecv_selected_features.csv")
    final_coef = pd.read_csv(extra_dir / "final_coef_or_table.csv")
    rfecv_features["Feature"] = rfecv_features["Feature"].map(clean_name)
    final_coef["Feature"] = final_coef["Feature"].map(clean_name)
    metrics_table_fmt = metrics_table.copy()
    for col in ["Accuracy", "F1", "AUC_mean", "AUC_std", "Sensitivity", "Specificity"]:
        metrics_table_fmt[col] = metrics_table_fmt[col].map(lambda x: f"{x:.4f}")
    auc_row = auc_ci[auc_ci["metric"] == "auc"].iloc[0]
    auc_point = auc_row["point_estimate"]
    auc_lower = auc_row["ci_lower"]
    auc_upper = auc_row["ci_upper"]
    dca_values_path = extra_dir / "dca_values.csv"
    if dca_values_path.exists():
        dca_df = pd.read_csv(dca_values_path)
        dca_min = float(dca_df["threshold"].min())
        dca_max = float(dca_df["threshold"].max())
    else:
        dca_min = 0.05
        dca_max = 0.60
    lines = []
    lines.append("# RESULTS_DIGEST")
    lines.append("")
    lines.append("**Data Facts**")
    lines.append(f"- n = {data_summary['total_samples']}")
    lines.append(f"Source: {legacy_dir.as_posix()}/data_summary.json")
    lines.append(f"- events = {event_rate['events']}")
    lines.append(f"Source: {extra_dir.as_posix()}/event_rate.json")
    lines.append(f"- event rate = {event_rate['event_rate']:.6f}")
    lines.append(f"Source: {extra_dir.as_posix()}/event_rate.json")
    lines.append(f"- original features = {data_summary['original_features']} (B-AS, 67 features)")
    lines.append(f"Source: {legacy_dir.as_posix()}/data_summary.json")
    lines.append(f"- RFECV selected features = {data_summary['selected_features']}")
    lines.append(f"Source: {legacy_dir.as_posix()}/data_summary.json")
    lines.append("")
    lines.append("**Table 1 (Five Models)**")
    lines.append("")
    lines.append("| Model | Accuracy | F1 | AUC_mean | AUC_std | Sensitivity | Specificity |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in metrics_table_fmt.iterrows():
        lines.append(
            f"| {row['Model']} | {row['Accuracy']} | {row['F1']} | {row['AUC_mean']} | {row['AUC_std']} | {row['Sensitivity']} | {row['Specificity']} |"
        )
    lines.append("")
    lines.append(f"Source: {legacy_dir.as_posix()}/metrics_table1.csv")
    lines.append("")
    lines.append("**AUC 95% CI (OOF Bootstrap)**")
    lines.append(f"- AUC = {auc_point:.4f}, 95% CI = [{auc_lower:.4f}, {auc_upper:.4f}] (bootstrap n=2000)")
    lines.append(f"Source: {extra_dir.as_posix()}/auc_ci.csv")
    lines.append("")
    lines.append("**Calibration**")
    lines.append(f"- Brier score = {calib_metrics['brier_score']:.6f}")
    lines.append(f"Source: {extra_dir.as_posix()}/calibration_metrics.json")
    lines.append(f"- Calibration intercept = {calib_metrics['calibration_intercept']:.6f}")
    lines.append(f"Source: {extra_dir.as_posix()}/calibration_metrics.json")
    lines.append(f"- Calibration slope = {calib_metrics['calibration_slope']:.6f}")
    lines.append(f"Source: {extra_dir.as_posix()}/calibration_metrics.json")
    lines.append("")
    lines.append("**DCA (Decision Curve Analysis)**")
    lines.append(f"- Threshold range used: {dca_min:.2f}-{dca_max:.2f}")
    lines.append(f"Source: {extra_dir.as_posix()}/dca_values.csv")
    lines.append("")
    lines.append("**RFECV Selected Features (17)**")
    lines.append("")
    lines.append("| # | Feature |")
    lines.append("| --- | --- |")
    for idx, feat in enumerate(rfecv_features["Feature"].tolist(), 1):
        lines.append(f"| {idx} | {feat} |")
    lines.append("")
    lines.append(f"Source: {legacy_dir.as_posix()}/rfecv_selected_features.csv")
    lines.append("")
    lines.append("**Final Coef/OR Table (17 features, full)**")
    lines.append("")
    lines.append("| Feature | Beta | SE | OR | OR_95CI_lower | OR_95CI_upper | P_value |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in final_coef.iterrows():
        lines.append(
            f"| {row['Feature']} | {row['Beta']:.6f} | {row['SE']:.6f} | {row['OR']:.6f} | {row['OR_95CI_lower']:.6f} | {row['OR_95CI_upper']:.6f} | {row['P_value']:.6f} |"
        )
    lines.append("")
    lines.append(f"Source: {extra_dir.as_posix()}/final_coef_or_table.csv")
    lines.append("")
    lines.append("**Main Metric Conventions**")
    lines.append("- Table 1 AUC uses fold mean +/- std from 5-fold CV.")
    lines.append(f"Source: {legacy_dir.as_posix()}/metrics_table1.csv")
    lines.append("- AUC 95% CI uses pooled OOF bootstrap (not the same as fold mean+/-std), so the two AUC summaries differ by convention.")
    lines.append(f"Source: {extra_dir.as_posix()}/auc_ci.csv")
    lines.append("")
    (pack_dir / "RESULTS_DIGEST.md").write_text("\n".join(lines), encoding="utf-8")
def main():
    parser = argparse.ArgumentParser(description="Reproduce legacy results + extra pack + paper pack.")
    parser.add_argument("--run-id", default="", help="Use existing run_id; if empty, run legacy rerun to create a new one.")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip running legacy_rerun.py.")
    args = parser.parse_args()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id.strip()
    if not run_id and not args.skip_legacy:
        before = list_run_dirs(results_dir)
        run_cmd([sys.executable, "src/legacy_rerun.py"])
        after = list_run_dirs(results_dir)
        new_run = detect_new_run(before, after)
        if not new_run:
            raise RuntimeError("Failed to detect new run directory after legacy_rerun.")
        run_id = new_run.replace("run_", "")
    elif not run_id and args.skip_legacy:
        raise ValueError("Either provide --run-id or allow running legacy_rerun.")
    run_cmd([sys.executable, "src/09_export_oof_predictions.py", "--run-id", run_id])
    run_cmd([sys.executable, "src/10_make_baseline_table.py", "--run-id", run_id, "--summary", "mean_sd"])
    run_cmd([sys.executable, "src/11_bootstrap_ci_from_oof.py", "--run-id", run_id])
    run_cmd([sys.executable, "src/12_calibration_pack.py", "--run-id", run_id])
    run_cmd([sys.executable, "src/13_dca_pack.py", "--run-id", run_id])
    run_cmd([sys.executable, "src/14_refit_final_model_or_forest.py", "--run-id", run_id])
    run_cmd([sys.executable, "src/15_patch_values_fill.py", "--run-id", run_id])
    make_paper_pack(run_id)
    print(f"Done. run_id={run_id}")
if __name__ == "__main__":
    main()