# Pediatric one-lung ventilation oxygenation model

This repository contains the analysis code for a retrospective cohort study of early oxygenation impairment after pediatric one-lung ventilation (OLV).

The primary analysis uses ridge logistic regression with ten candidate predictors available 30 minutes after OLV initiation. Elastic-net and LASSO logistic regression are included as secondary comparisons. Model performance is estimated with repeated nested cross-validation. Imputation, standardization, and hyperparameter tuning are fitted within the training portion of each resampling split.

## Repository contents

- `run_primary_analysis.py`: primary repeated nested cross-validation analysis, model comparisons, calibration, threshold summaries, and figures.
- `run_sensitivity_analyses.py`: iterative-imputation, complete-case, and bootstrap optimism-correction analyses.
- `requirements.txt`: Python package versions used for the analysis.
- `results/`: aggregate reference results and figures from the reported analysis.

## Data

The patient-level workbook is not included in this public release. To rerun the
analysis, place an appropriately approved analysis dataset at:

```text
data/original_data_p6e.xlsx
```

The archived reference results exclude patient-level out-of-fold prediction
files. Local analysis runs generate those files in the selected output
directory; do not redistribute them without the applicable data-sharing
approval.

## Environment

Python 3.9--3.11 is recommended; the reported analysis used Python 3.9.6.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the analyses

Primary analysis:

```bash
python run_primary_analysis.py
```

Sensitivity analyses:

```bash
python run_sensitivity_analyses.py
```

Default outputs are written to `results/primary` and `results/sensitivity`. Alternative paths can be provided explicitly:

```bash
python run_primary_analysis.py --input path/to/data.xlsx --output path/to/primary_results
python run_sensitivity_analyses.py --input path/to/data.xlsx --output path/to/sensitivity_results
```
