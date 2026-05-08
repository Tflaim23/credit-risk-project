# Thomas Flaim Credit Risk PD Modeling on LendingClub Data

An end to end credit risk project built in Python on LendingClub loan level data. This is my second project and starts with data ingestion and leakage control, moves through a modeling and probability evaluation, then turns model outputs into a credit policy with expected loss, stress testing and more analysis. Similar to my first gentrification risk project I wanted to build technical skills and knowledge by attempting a real indivdual workflow, rather than just doing practice and examples. The project is built around real risk work:

- **Leakage-safe forward-in-time data split**
- **Probability quality**, not just ranking metrics
- **Calibration tested**
- **Policy Decision** using `EL = PD × LGD × EAD`
- **Vintage analysis**
- **Monte Carlo and deterministic fixed-portfolio stress testing**


## Final project summary

I built a full credit risk pipeline on LendingClub data using a leakage-safe forward-in-time split. First I began with EDA and preprocessing of the data to ensure it was clean and ready to be modeled. I started with a logistic regression baseline and then trained XGBoost, which outperformed logistic on the test set and became the chosen model. I tested both sigmoid and isotonic calibration, but neither improved final test log loss or Brier score, so I kept the uncalibrated XGBoost and documented the result. I then selected a PD threshold of `0.15` using policy curves, evaluated the accepted portfolio across vintages (times), identified maturity bias in the latest vintages, and finished with deterministic and Monte Carlo fixed-portfolio stress testing. Througout this process all work has been my own with the help of AI and asking questions to people with knowledge of the topics such as Dr. Tao Pang. I was able to develop these skills throughout my time working on this project:

- **Large-scale data handling:** Learned how to work with messy, high-volume datasets and turn them into clean modeling inputs

- **Leakage-aware research design:** Learned how to structure train/test splits and validation windows so results better reflect realistic out-of-sample performance

- **Feature engineering and preprocessing:** Learned how to build repeatable preprocessing steps for imputation/interpolation, encoding, scaling, missingness handling, and model-ready matrix creation

- **Predictive modeling:** Learned how to train, compare, and interpret baseline and nonlinear models, including logistic regression and boosted trees

- **Probability-focused evaluation:** Learned how to evaluate models with ranking, calibration, and probability-quality metrics rather than relying only on accuracy

- **Risk translation:** Learned how to turn model probabilities into financial decision metrics using expected loss, exposure, and loss assumptions

- **Portfolio stress testing:** Learned how to test how a fixed portfolio behaves under deterministic and Monte Carlo downside scenarios

- **Reproducible quantitative workflow:** Learned how to organize scripts, configs, artifacts, tables, plots, and documentation into a project that can be reviewed and rerun

- **Technical skils:** - Learned python data analysis with `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`, and YAML configs so I can reuse untouched source files with different inputs

---

## Realistic limitations

- resolved-only target definition creates maturity bias in more recent data
- LGD was assumed constant rather than estimated
- EAD was proxied by `loan_amnt`
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
- `notebooks/` just one eda notebook
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