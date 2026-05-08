# Thomas Flaim Credit Risk PD Modeling on LendingClub Data

An end to end credit risk project built in Python on LendingClub loan level data. This project starts with data ingestion and leakage control, moves through modeling and probability evaluation, then turns model outputs into a credit policy with expected loss, stress testing and more analysis. I wanted to build technical skills and knowledge by attempting a real individual workflow, rather than just doing practice and examples. The project is built around real risk work:

- **Leakage-safe forward-in-time data split**
- **Probability quality**, not just ranking metrics
- **Calibration tested**
- **Policy decision** using `EL = PD × LGD × EAD`
- **Vintage analysis**
- **Monte Carlo and deterministic fixed-portfolio stress testing**


## Final project summary

I built a full credit risk pipeline on LendingClub data using a leakage-safe forward-in-time split. I began with EDA and preprocessing of the data to make sure it was clean and ready to be modeled. I first trained a logistic regression baseline and then trained XGBoost, which outperformed logistic on the test set and became the chosen model. I tested both sigmoid and isotonic calibration, but neither improved final test log loss or Brier score, so I kept the uncalibrated XGBoost. I then selected a PD threshold of `0.15` using policy curves, evaluated the accepted portfolio across vintages (times), identified maturity bias in the latest vintages, and finished with deterministic and Monte Carlo fixed-portfolio stress testing. I was able to develop these skills throughout my time working on this project:

- **Large-scale data handling:** Worked with a messy dataset with over 1,000,000 rows and turned it into a clean modeling input

- **Leakage-aware research design:** Structured train/test splits and validation windows so results better reflect realistic out-of-sample performance

- **Feature engineering and preprocessing:** Built repeatable preprocessing steps for imputation, missingness handling, and model-ready matrix creation

- **Predictive modeling:** Learned how to train, compare, and interpret baseline and nonlinear models, in this project being logistic regression and boosted trees

- **Probability-focused evaluation:** Evaluated models with ranking and probability-quality metrics rather than relying only on accuracy

- **Risk translation:** Turned model probabilities into financial decision metrics using expected loss, exposure, and loss assumptions

- **Portfolio stress testing:** Tested how a fixed portfolio behaves under deterministic and Monte Carlo downside scenarios

- **Reproducible quantitative workflow:** Organized scripts, configs, artifacts, tables, plots, and documentation into a project that can be reviewed

- **Technical skills:** Learned python data analysis with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`, and YAML configs so I can reuse untouched source files with different inputs

---

## Realistic limitations

- resolved-only target definition creates maturity bias in more recent data, this was acknowledged but unavoidable
- LGD was assumed constant rather than estimated
- EAD was proxied by `loan_amnt` which would not be 1-1 in a real scenario, especially considering partially paid off outcomes
- stress multipliers were scenario assumptions, not macroeconometric forecasts
- calibration did not generalize well out of time, so the raw model was retained with that limitation documented

---

## Repo structure

- `src/` scripts
- `configs/` YAML config files
- `data/raw/` raw data loan.csv from LendingClub, gitignored
- `data/processed/` processed outputs, gitignored
- `data/sample/` small committed sample
- `reports/tables/` CSV outputs
- `reports/figures/` PNG outputs
- `docs/` data dictionary and notes
- `notebooks/` EDA notebook
- `requirements/` all packages needed
---

## How to reproduce

### Main environment

- WSL Ubuntu
- VS Code
- local `.venv`

### Typical commands in terminal

- source .venv/bin/activate
- python src/ScriptName.py --config configs/ConfigName_v1.yaml

--- 


# Below will be an in-depth and step by step description of my process throughout the project. This is not necessary to read through in its entirety, it is simply for full documentation of how and what I did in my time working on this project



## Dataset

### Source and population

- Source data: LendingClub
- https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data
- Modeling population: **resolved loans only**
- Target:
  - `y = 1` bad = `Charged Off` + `Does not meet credit policy Charged Off`
  - `y = 0` good = `Fully Paid` + `Does not meet credit policy Fully Paid`

### Modeling dataset facts

- Rows: **1,306,356**
- Issue date range: **2007-06 to 2018-12**
- Target mix:
  - good: **79.9124%**
  - bad: **20.0876%**
- Final selected raw feature count: **85**
- Final preprocessed model input columns: **276**

The raw feature set contains 85 selected LendingClub variables. After preprocessing, one-hot encoding of categorical variables (splitting into 0/1 binaries rather than one col) expanded the final model input matrix to 276 columns


**Sources:** [`src/make_modeling_dataset.py`](src/make_modeling_dataset.py), [`configs/lendingclub_modeling.yaml`](configs/lendingclub_modeling.yaml), [`reports/tables/modeling_v1/data_profile.csv`](reports/tables/modeling_v1/data_profile.csv), [`reports/tables/modeling_v1/y_counts.csv`](reports/tables/modeling_v1/y_counts.csv), [`reports/tables/loan_status_counts.csv`](reports/tables/loan_status_counts.csv)

### Why unresolved loans were dropped

I removed unresolved statuses `Current`, `Late`, and `In Grace Period` because they do not have final outcomes yet. Treating unresolved loans as good would make the measured performance look better than it really is

**Sources:** [`src/make_modeling_dataset.py`](src/make_modeling_dataset.py), [`configs/lendingclub_modeling.yaml`](configs/lendingclub_modeling.yaml), [`reports/tables/loan_status_counts.csv`](reports/tables/loan_status_counts.csv)

---

## Beginning approach

Before running the full LendingClub workflow, I used a small fake sample dataset to test my initial scripts. That let me debug the pipeline before potentially messing up the real dataset.

**Sources:** [`src/create_sample_data.py`](src/create_sample_data.py)[`data/sample/loans_sample.csv`](data/sample/loans_sample.csv), [`src/load.py`](src/load.py) [`configs/sample_load.yaml`](configs/sample_load.yaml)

---

## Project workflow

1. Load raw CSV and convert to parquet
2. Profile the data for shape, missingness, duplicates, cardinality, and date coverage
3. Create the resolved-only modeling dataset and binary target
4. Run a leakage audit and freeze the feature set
5. Split forward in time using `issue_d`
6. Build preprocessing with imputation, missing flags, one hot encoding, and numeric scaling
7. Train logistic regression baseline
8. Train XGBoost comparison model
9. Test threshold refinement and calibration methods
10. Build policy curves and select a PD threshold
11. Check stability over time at the chosen threshold
12. Run deterministic and Monte Carlo fixed-portfolio stress tests

---

## Data quality, leakage control, and preprocessing

### Profiling and cleaning

I profiled the modeling dataset for:
- row and column count
- memory usage
- duplicates
- date range
- missingness
- cardinality

I also created a written cleaning contract describing missingness handling, leakage controls, and final feature selection

**Sources:** [`src/profile.py`](src/profile.py), [`configs/lendingclub_modeling_profile.yaml`](configs/lendingclub_modeling_profile.yaml), [`reports/tables/modeling_v1/data_profile.csv`](reports/tables/modeling_v1/data_profile.csv), [`reports/tables/modeling_v1/missingness.csv`](reports/tables/modeling_v1/missingness.csv), [`docs/model_inputs_cleaning_contract.md`](docs/model_inputs_cleaning_contract.md)

### Leakage audit and feature freeze

I used a leakage audit to flag:
- columns with high missingness
- high-cardinality text fields
- likely post-origination leakage fields
- manual drop or keep overrides

Final predictor set: **85 features**

**Sources:** [`src/leakage_audit_v1.py`](src/leakage_audit_v1.py), [`configs/leakage_audit_v1.yaml`](configs/leakage_audit_v1.yaml), [`reports/tables/modeling_v1/drop_columns_v1.csv`](reports/tables/modeling_v1/drop_columns_v1.csv), [`reports/tables/modeling_v1/keep_columns_v1.csv`](reports/tables/modeling_v1/keep_columns_v1.csv), [`src/make_feature_set_v1.py`](src/make_feature_set_v1.py), [`configs/feature_set_v1.yaml`](configs/feature_set_v1.yaml), [`reports/tables/modeling_v1/feature_set_v1.csv`](reports/tables/modeling_v1/feature_set_v1.csv)

### Forward-in-time split

I split the modeling dataset by date so earlier loans trained the models and later loans evaluated them

- Train rows: **1,029,030**
- Test rows: **277,326**
- Train window: **2007-06 to 2016-08**
- Test window: **2016-09 to 2018-12**

**Sources:** [`src/split.py`](src/split.py), [`configs/split_v1.yaml`](configs/split_v1.yaml), [`reports/tables/modeling_v1/split_manifest_v1.csv`](reports/tables/modeling_v1/split_manifest_v1.csv)

### Preprocessing

Preprocessing was fit on train only and then applied blindly to test

Main preprocessing choices:
- numeric median imputation
- numeric missing indicator flags
- most-frequent categorical imputation
- one hot encoding for categoricals
- numeric scaling in the refined workflow

**Sources:** [`src/preprocess.py`](src/preprocess.py), [`configs/preprocess_v2.yaml`](configs/preprocess_v2.yaml), [`reports/tables/modeling_v1/preprocess_manifest_v2.csv`](reports/tables/modeling_v1/preprocess_manifest_v2.csv), [`reports/tables/modeling_v1/preprocessed_feature_names_v2.csv`](reports/tables/modeling_v1/preprocessed_feature_names_v2.csv)

---

## EDA

I saved a few EDA plots for the modeling population:

- bad rate by grade
- loan amount distribution
- DTI distribution

**Sources:** [`notebooks/01_eda_modeling_v1.ipynb`](notebooks/01_eda_modeling_v1.ipynb), [`reports/figures/modeling_v1/bad_rate_by_grade.png`](reports/figures/modeling_v1/bad_rate_by_grade.png), [`reports/figures/modeling_v1/loan_amnt_hist.png`](reports/figures/modeling_v1/loan_amnt_hist.png), [`reports/figures/modeling_v1/dti_hist.png`](reports/figures/modeling_v1/dti_hist.png)

---

## Modeling

### Logistic regression baseline

Logistic regression was the baseline

**Refined logistic regression test metrics**
- ROC AUC: **0.70556** Measures whether the model ranks bad loans as riskier than good loans across thresholds 1 is perfect, 0.5 basline
- PR AUC: **0.38416** Measures how well the model identifies the bad-loan class specifically 1 is perfect, bad rate is the baseline 
- Log loss: **0.48282** Measures the quality of the predicted probabilities, penalizes confident wrong predictions 0 is perfect 
- Brier score: **0.15591** Essentially MSE, 0 is perfect
- Accuracy is less meaningful because it can just predict good across the board and do fairly well

**Sources:** [`src/train_logreg.py`](src/train_logreg.py), [`configs/train_logreg_v2.yaml`](configs/train_logreg_v2.yaml), [`reports/tables/modeling_v1/logreg_metrics_v2.csv`](reports/tables/modeling_v1/logreg_metrics_v2.csv), [`reports/tables/modeling_v1/logreg_coefficients_v2.csv`](reports/tables/modeling_v1/logreg_coefficients_v2.csv)

### XGBoost comparison model

XGBoost outperformed logistic on the main forward-in-time test set and became the champion model for the project

**Main XGBoost comparison test metrics**
- ROC AUC: **0.71972** Better
- PR AUC: **0.40890** Better
- Log loss: **0.47259** Better
- Brier score: **0.15332** Better

**Sources:** [`src/train_xgboost.py`](src/train_xgboost.py), [`configs/train_xgboost_v1.yaml`](configs/train_xgboost_v1.yaml), [`reports/tables/modeling_v1/xgboost_metrics_v1.csv`](reports/tables/modeling_v1/xgboost_metrics_v1.csv), [`reports/tables/modeling_v1/xgboost_feature_importances_v1.csv`](reports/tables/modeling_v1/xgboost_feature_importances_v1.csv), [`src/model_comparison.py`](src/model_comparison.py), [`configs/model_comparison_v1.yaml`](configs/model_comparison_v1.yaml), [`reports/tables/modeling_v1/model_comparison_v1.csv`](reports/tables/modeling_v1/model_comparison_v1.csv)

### Threshold refinement

I also checked whether the model ranking conclusion depended on the default 0.50 classification cutoff by sweeping thresholds and choosing train-based F1 thresholds

- XGBoost still beat logistic after threshold refinement

**Sources:** [`src/threshold_sweep.py`](src/threshold_sweep.py), [`configs/threshold_sweep_logreg_v1.yaml`](configs/threshold_sweep_logreg_v1.yaml), [`configs/threshold_sweep_xgboost_v1.yaml`](configs/threshold_sweep_xgboost_v1.yaml), [`reports/tables/modeling_v1/logreg_threshold_summary_v1.csv`](reports/tables/modeling_v1/logreg_threshold_summary_v1.csv), [`reports/tables/modeling_v1/xgboost_threshold_summary_v1.csv`](reports/tables/modeling_v1/xgboost_threshold_summary_v1.csv)

---

## Calibration results

I built a clean calibration workflow by splitting the training period again into:

- subtrain: **825,575 rows**
- calibration: **203,455 rows**

This kept model fitting, calibration, and final testing separated in time

**Sources:** [`src/split_calibration.py`](src/split_calibration.py), [`configs/calibration_split_v1.yaml`](configs/calibration_split_v1.yaml), [`reports/tables/modeling_v1/calibration_split_manifest_v1.csv`](reports/tables/modeling_v1/calibration_split_manifest_v1.csv), [`src/preprocess_calibration.py`](src/preprocess_calibration.py), [`configs/preprocess_calibration_v1.yaml`](configs/preprocess_calibration_v1.yaml), [`reports/tables/modeling_v1/preprocess_calibration_manifest_v1.csv`](reports/tables/modeling_v1/preprocess_calibration_manifest_v1.csv)

Then I tested both sigmoid and isotonic calibration on a calibration slice

**Conclusion:** both methods improved fit on the calibration slice itself, but neither improved the final untouched test set relative to the uncalibrated XGBoost model. In particular, neither improved final test log loss or Brier score

Because of that, I kept the **uncalibrated XGBoost** as the probability model for policy analysis

**Sources:** [`src/calibrate_xgboost.py`](src/calibrate_xgboost.py), [`configs/calibrate_xgboost_v1.yaml`](configs/calibrate_xgboost_v1.yaml), [`reports/tables/modeling_v1/xgboost_subtrain_metrics_v1.csv`](reports/tables/modeling_v1/xgboost_subtrain_metrics_v1.csv), [`reports/tables/modeling_v1/xgboost_calibration_metrics_v1.csv`](reports/tables/modeling_v1/xgboost_calibration_metrics_v1.csv), [`reports/figures/modeling_v1/xgboost_calibration_curve_test_v1.png`](reports/figures/modeling_v1/xgboost_calibration_curve_test_v1.png)

---

## Policy layer

### Expected loss framework

The policy layer uses:

`EL = PD × LGD × EAD`

where:
- `PD` = predicted probability of default from the uncalibrated XGBoost model
- `LGD` = assumed constant loss given default
- `EAD` = exposure at default

For the main policy analysis, I used:
- `LGD = 0.40`
- `EAD = loan_amnt`

**Sources:** [`src/make_policy_predictions.py`](src/make_policy_predictions.py), [`configs/make_policy_predictions_v2.yaml`](configs/make_policy_predictions_v2.yaml), [`reports/tables/modeling_v1/policy_inputs_manifest_v2.csv`](reports/tables/modeling_v1/policy_inputs_manifest_v2.csv), [`src/policy.py`](src/policy.py), [`configs/policy_v1.yaml`](configs/policy_v1.yaml)

### Threshold selection

I selected a PD threshold of **0.15** after looking at the policy curves and choosing the point where the tradeoff between acceptance and risk looked most reasonable on the test set

The threshold was not based on a fixed rule, but rather my own discretion. I wanted a threshold that was not too restrictive, but also did not let expected loss rise too quickly

This threshold means:
- accept loans with `predicted_prob_bad <= 0.15`
- reject loans with `predicted_prob_bad > 0.15`

At this threshold, test acceptance was about **43.26%**.

**Sources:** [`src/policy.py`](src/policy.py), [`configs/policy_v1.yaml`](configs/policy_v1.yaml), [`reports/tables/modeling_v1/policy_sweep_v1.csv`](reports/tables/modeling_v1/policy_sweep_v1.csv), [`src/plot_policy_curves.py`](src/plot_policy_curves.py), [`reports/figures/modeling_v1/policy_expected_loss_rate_by_threshold_v1.png`](reports/figures/modeling_v1/policy_expected_loss_rate_by_threshold_v1.png), [`reports/figures/modeling_v1/policy_expected_loss_by_threshold_v1.png`](reports/figures/modeling_v1/policy_expected_loss_by_threshold_v1.png), [`docs/policy_threshold_choice_v1.md`](docs/policy_threshold_choice_v1.md)


## Vintage stability

I evaluated the chosen policy threshold across test vintages (times of origination).

Stability analysis showed that the last two vintages looked unrealistically good. After reviewing the results, I concluded that the issue was **label maturity bias** caused by using a resolved-only dataset. The most recent vintages had many unresolved loans removed, leaving a smaller and cleaner subset.

I reran the main stability view excluding the final two immature vintages to get a more realistic result which showed fairly stable outcomes over time but showed the accepted portfolio still had some PD underestimation relative to realized bad rates as shown specifically in the risk chart.

**Sources:** [`src/vintage_policy_stability.py`](src/vintage_policy_stability.py), [`configs/vintage_policy_stability_v1.yaml`](configs/vintage_policy_stability_v1.yaml), [`configs/vintage_policy_stability_v2.yaml`](configs/vintage_policy_stability_v2.yaml), [`reports/tables/modeling_v1/vintage_policy_stability_v1.csv`](reports/tables/modeling_v1/vintage_policy_stability_v1.csv), [`reports/tables/modeling_v1/vintage_policy_stability_v2.csv`](reports/tables/modeling_v1/vintage_policy_stability_v2.csv), [`reports/figures/modeling_v1/vintage_acceptance_rate_v2.png`](reports/figures/modeling_v1/vintage_acceptance_rate_v2.png), [`reports/figures/modeling_v1/vintage_accepted_risk_v2.png`](reports/figures/modeling_v1/vintage_accepted_risk_v2.png)

---

## Deterministic fixed-portfolio stress test

I stress tested the **already accepted** portfolio at threshold `0.15`, rather than reapplying the acceptance threshold after stressing PDs

### Fixed accepted portfolio
- accepted loans: **119,985**
- fixed EAD: about **$1.468B**

### Deterministic results
- Base expected loss: **$49.68M**
- Conservative overlay expected loss: **$57.14M**
- Mild recession expected loss: **$81.98M**
- Severe recession expected loss: **$161.48M**

Base expected loss rate was **3.39%**, while the historical actual loss proxy on the same fixed portfolio was **4.26%**, consistent with a little underprediction in the accepted book.

Applying the multiplier from conservative overlay ended up being much closer to a true prediction of PD.

**Sources:** [`src/stress_test_fixed_portfolio.py`](src/stress_test_fixed_portfolio.py), [`configs/stress_test_fixed_portfolio_v1.yaml`](configs/stress_test_fixed_portfolio_v1.yaml), [`reports/tables/modeling_v1/stress_test_fixed_portfolio_v1.csv`](reports/tables/modeling_v1/stress_test_fixed_portfolio_v1.csv), [`reports/figures/modeling_v1/stress_fixed_portfolio_el_v1.png`](reports/figures/modeling_v1/stress_fixed_portfolio_el_v1.png), [`reports/figures/modeling_v1/stress_fixed_portfolio_el_rate_v1.png`](reports/figures/modeling_v1/stress_fixed_portfolio_el_rate_v1.png)

---

## Monte Carlo fixed-portfolio stress test

As a final analysis layer, I ran Monte Carlo stress overlays on the same fixed accepted portfolio

- **2,000 simulations per scenario**
- **4 scenarios**
- **8,000 total simulations**

### Monte Carlo summary

- Base: Mean EL $49.72M, P95 EL $54.45M, Mean Rate 3.39%, P95 Rate 3.71% 
- Conservative overlay: Mean EL $57.04M, P95 EL $62.22M, Mean Rate 3.89%, P95 Rate 4.24% 
- Mild recession: Mean EL $81.61M, P95 EL $94.34M, Mean Rate 5.56%, P95 Rate 6.43% 
- Severe recession: Mean EL $161.32M, P95 EL $192.65M, Mean Rate 10.99%, P95 Rate 13.13% 

The Monte Carlo means lined up closely with the deterministic stress test, while also showing the distribution and tail risk around each scenario

**Sources:** [`src/monte_carlo_fixed_portfolio.py`](src/monte_carlo_fixed_portfolio.py), [`configs/monte_carlo_fixed_portfolio_v1.yaml`](configs/monte_carlo_fixed_portfolio_v1.yaml), [`reports/tables/modeling_v1/monte_carlo_fixed_portfolio_summary_v1.csv`](reports/tables/modeling_v1/monte_carlo_fixed_portfolio_summary_v1.csv), [`reports/figures/modeling_v1/monte_carlo_fixed_portfolio_hist_v1.png`](reports/figures/modeling_v1/monte_carlo_fixed_portfolio_hist_v1.png)

---
# Final comments

- Similar to my first gentrification risk project, all analysis is my own. I used AI as a learning aid for package selection, script organization, debugging support, and general Python guidance.

- This project pushed me beyond simply fitting a model. I ran into many roadblocks, for example around calibration, stability, and training logreg, but I am proud of how I worked through them and turned the project into a complete end-to-end credit risk workflow.

- I plan to keep building on this work by improving my Python fluency and working on more advanced projects during my time in the NC State Master of Financial Mathematics program.

** for **bold**  backticks for `code/terms`