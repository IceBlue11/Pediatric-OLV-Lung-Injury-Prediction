# Primary analysis results

## Candidate predictors

Male sex; weight; preoperative CRP; preoperative neutrophils; driving pressure at OLV 30 min; PEEP at OLV 30 min; PaCO2 at OLV 30 min; blood base excess at OLV 30 min; heart rate at OLV 30 min; and Ti/Ttot at OLV 30 min. Age and intraoperative P/F ratio were not included.

Ti/Ttot and driving pressure were read directly from their existing source-workbook columns. Ti/Ttot was independently checked against inspiratory time (seconds) × respiratory rate (breaths/minute) / 60, and driving pressure was independently checked against Pplat - PEEP. The reconstructed values were not used to overwrite the recorded source values.

## Dataset checks

- Patients: 181
- Events: 52 (28.729%)
- Maximum absolute discrepancy for driving pressure versus Pplat - PEEP: 0
- Driving-pressure rows checked / mismatches: 172 / 0
- Maximum absolute discrepancy for Ti/Ttot versus Ti × RR / 60: 3.88578058619e-16
- Ti/Ttot rows checked / mismatches: 172 / 0
## Validation design

The primary model was ridge logistic regression. Elastic-net and LASSO were secondary comparisons. The outer validation was stratified 5-fold cross-validation repeated 10 times. Within each outer training fold, a separate stratified 5-fold inner loop performed median imputation, standardisation and hyperparameter tuning by minimum log loss. The untouched outer fold was used only for performance estimation.

## Primary ridge results

- Mean repeated nested-CV AUC: 0.686285 (SD 0.020387)
- Mean repeated nested-CV AUPRC: 0.486012
- Mean repeated nested-CV Brier score: 0.186445
- Mean calibration intercept: -0.045305
- Mean calibration slope: 0.941553

The resampling percentiles in the performance summary describe variability across the 10 repeated partitions; they are not an external-validation confidence interval. The bootstrap CI file resamples fixed mean repeated OOF predictions and does not repeat model development.

## Generated output files

- `candidate_variable_definitions_and_missingness.csv`: exact source columns, units, derivation and missingness.
- `candidate_descriptive_statistics.csv`: distributions overall and by outcome.
- `candidate_spearman_correlation.csv` and `candidate_vif.csv`: collinearity checks.
- `inner_tuning_results.csv`: every tested hyperparameter combination in every outer training set.
- `selected_hyperparameters_by_outer_fold.csv`: the selected setting for each outer fold.
- `performance_by_outer_fold.csv` and `performance_by_repeat.csv`: granular validation results.
- `model_performance_summary.csv`: principal summary; report the mean across repeats.
- `repeated_nested_oof_predictions.csv`: all 10 predictions per patient per model; generated locally but omitted from the public reference-results archive.
- `mean_repeated_oof_predictions.csv`: patient-level mean of the 10 OOF predictions; generated locally but omitted from the public reference-results archive.
- `selection_stability.csv`: descriptive elastic-net/LASSO selection frequencies.
- `final_model_coefficients_descriptive.csv`: post-development full-cohort fit for the potential equation; not an unbiased performance estimate and not for conventional p-values.
