# An Explainable Machine Learning Approach to Human Papillomavirus Vaccination Uptake Among Adolescent Girls in Ethiopia


Code for the manuscript "An Explainable Machine Learning Approach to Human Papillomavirus Vaccination Uptake Among Adolescent Girls in Ethiopia" (Intelligence-Based Medicine). The analysis uses girls aged 15 to 17 years in the 2024-25 Ethiopia Demographic and Health Survey (EDHS; 2,842 girls with a reported vaccination status).

## Data (not included)
DHS data cannot be redistributed. Register and request the Ethiopia 2024-25 individual recode (`ETIR8AFL`) at https://dhsprogram.com/data/. 

## Run
Python 3.10 or later: `pip install -r requirements.txt`, then `python hpv_vaccination_uptake_ml.py`.
Set `RUN_NESTED_CV = False` at the top of the script to skip the slow nested cross-validation (5 repeats x 20 search draws) and reuse the `nested_cv_*.csv` files already in `data/`.

## Pipeline (one file, `hpv_vaccination_uptake_ml.py`)
| Function | Purpose |
|---|---|
| `run_build_design` | Weights, PSU, strata from the IR file; alignment check |
| `run_eda` | Bar panels, Cramer's V matrix, chi-square, VIF |
| `run_survey_prevalence` | Weighted prevalence, design effect, regional shares, "don't know" sensitivity, vaccination card |
| `run_preprocessing_lasso` | 80/20 development split, LASSO with the 1-SE rule |
| `run_table1_characteristics` | Table 1, unweighted and survey-weighted |
| `run_model_comparison`, `run_hyperparameter_tuning`, `run_confusion_matrices_tuned` | Six models on the development split |
| `run_lr_odds_ratios` | Development-split odds ratios; survey-weighted 29-feature model, all 14 regions, wealth sensitivity, forest plot |
| `run_shap_analysis` | SHAP for logistic regression and XGBoost |
| `run_nested_cv`, `run_nested_cv_summary` | Nested repeated 5x5 CV, Nadeau-Bengio paired tests, LASSO stability |
| `run_calibration_analysis`, `run_decision_curve_analysis` | Calibration slope/intercept and decision curves (event = unvaccinated) from out-of-fold predictions |

## Main outputs (written to `data/`)
| File | Content |
|---|---|
| `table1_sample_characteristics.csv/.tex`, `descriptive_weighted.csv` | Table 1, unweighted and survey-weighted |
| `region_weighted_prevalence.csv`, `region_weighted.png` | Weighted regional uptake and population shares |
| `lr29_unw_cluster_svy.csv`, `region_all14_weighted.csv`, `wealth_sensitivity_weighted.csv` | Survey-weighted models, 14-region contrasts, wealth sensitivity |
| `final_svy_model_or.csv`, `svy_forest_plot.png` | Final design-based odds ratios and forest plot |
| `nested_cv_metrics.csv`, `nested_cv_summary.csv`, `nested_cv_paired_tests.csv`, `lasso_selection_stability.csv`, `nested_cv_auc.png` | Nested cross-validation results |
| `calibration_pooled.csv`, `calibration_oof.png`, `dca_unvaccinated.csv`, `dca_unvaccinated.png` | Calibration and decision curves |
| `cramers_v_matrix.csv/.png`, `shap_*.png`, `confusion_matrices_tuned.png` | Association matrix, SHAP and confusion-matrix figures |

## Design
Strata = `v022`, PSU = `v021`, weight = `v005/1e6`; stratified-cluster (with-replacement) Taylor linearization, t degrees of freedom = 753 - 27 = 726. Random seeds are fixed throughout.

## Results
`results/` holds the aggregate tables reported in the manuscript (survey-weighted odds ratios, regional contrasts, nested CV summaries and fold-level metrics, calibration, decision curves, LASSO selection stability, weighted descriptive table).

## Reproducibility note
Logistic-regression, survey-weighted and benchmark results reproduce exactly. Tree-based models (XGBoost, LightGBM, CatBoost, random forest) and the perceptron can differ in the second or third decimal across package versions and CPU thread counts, even with fixed seeds. The values reported in the manuscript are those in `results/`.

## Region codes
1 Tigray, 2 Afar, 3 Amhara, 4 Oromia, 5 Somali, 6 Benishangul-Gumuz, 7 Central Ethiopia, 8 Sidama, 9 South West Ethiopia, 10 South Ethiopia, 12 Gambella, 13 Harari, 14 Addis Ababa, 15 Dire Dawa.

