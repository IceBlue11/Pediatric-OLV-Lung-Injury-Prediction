# Pediatric OLV Lung Injury Prediction

This repository contains the code used to rerun the pediatric one-lung ventilation lung injury prediction analyses.

## Files

- `run_full_pipeline.py`: end-to-end entry point for the rerun pipeline.
- `train_rfecv_lr_model.py`: RFECV logistic regression training and comparison workflow.
- `model_comparison_analysis.py`: six-model comparison analysis.
- `model_subset_analysis.py`: compact predictor subset analysis.
- `src/`: numbered post-processing and packaging steps used by the rerun pipeline.

## Usage

Run the full pipeline from the repository root:

```bash
python3 run_full_pipeline.py
```

Input data is expected under `data/raw/` and is intentionally excluded from version control.
