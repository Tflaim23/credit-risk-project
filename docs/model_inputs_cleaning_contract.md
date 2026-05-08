# Modeling Inputs Cleaning Contract

## Overall order of what I did before modeling
1. Load: converted the raw LendingClub csv to parquet for better processing (`src/load.py`)
2. Profile: looked at missingness, cardinality, duplicates, and date coverage (`src/profile.py`)
3. Create modeling dataset and target: filtered to resolved loans only and created the binary target `y` (`src/make_modeling_dataset.py`)
4. Leakage audit: reviewed columns and flagged drops for high missingness, highly uniquw text, and likely leakage (`src/leakage_audit_v1.py`)
5. Freeze feature set: removed `y` and `issue_d` from the keep list to produce the final predictor list (`src/make_feature_set_v1.py`)
6. Split: performed a forward in time 80/20 split so earlier loans train the model and later loans evaluate it (`src/split.py`)
7. Preprocess: fit imputation, missing flags, encoding, and later numeric scaling on train only, then applied the same transformer to test blindly (`src/preprocess.py`)

## Dataset
- Source: LendingClub loan.csv converted to Parquet for efficient processing
- Modeling dataset: resolved outcomes only, with target `y` (1=bad, 0=good)
- Rows: 1,306,356
- Date range (issue_d): 2007-06 to 2018-12
- Bad rate: ~0.2009

## Dropped unresolved statuses
Loans without a clear final outcome were removed to ensure target labels are trustworthy:
- Current
- Late (16-30 days)
- Late (31-120 days)
- In Grace Period
- Default

## EDA artifacts generated
- Bad rate by grade: `reports/figures/modeling_v1/bad_rate_by_grade.png`
- Loan amount distribution: `reports/figures/modeling_v1/loan_amnt_hist.png`
- DTI distribution: `reports/figures/modeling_v1/dti_hist.png`

## Missingness handling
- Full missingness report: `reports/tables/modeling_v1/missingness.csv`
- Columns with >95% missing were dropped (`drop_columns_v1.csv`)
- These included hardship, settlement, and joint application columns which were largely unpopulated
- Remaining columns with partial missingness are handled in preprocessing time via median imputation for numeric columns and most-frequent imputation for categorical columns
- A binary missing indicator flag for each numeric column that had any missingness in the training set

## Column selection
- Full audit: `reports/tables/modeling_v1/drop_columns_v1.csv`
- Drop reasons: HIGH_MISSING, HIGH_CARDINALITY, LEAKAGE_NAME_PATTERN, MUST_DROP
- Final predictor set: `reports/tables/modeling_v1/feature_set_v1.csv` (85 features)
- Post-origination leakage columns (recoveries, collection fees, etc.) were explicitly excluded