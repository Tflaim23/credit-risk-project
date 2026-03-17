# Modeling Inputs Cleaning Contract

## Dataset
- Source: LendingClub loan.csv which I changed to parquet
- Modeling dataset: resolved outcomes only, with target `y` (1=bad, 0=good)
- Rows: 1,306,356
- Date range (issue_d): 2007-06 to 2018-12
- Bad rate: ~0.2009

- Dropped unresolved statuses:
  - Current
  - Late
  - Grace Period
  - Default


## EDA artifacts generated
- Bad rate by grade: `reports/figures/modeling_v1/bad_rate_by_grade.png`
- Loan amount distribution: `reports/figures/modeling_v1/loan_amnt_hist.png`
- DTI distribution: `reports/figures/modeling_v1/dti_hist.png`

## Missingness snapshot ^^^^^^^^^^^^^^^NOT DONE^^^^^^^^^^^^^^^^
- See: `reports/tables/modeling_v1/missingness.csv`
- many hardship and settlement columns are >99% missing which will be handled