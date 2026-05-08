# Policy Threshold Choice Document 

## Selected PD threshold

Chosen threshold: 0.15

## Why I picked 0.15 

Within a risk team there would likely be rules as to what risk the company can tolerate, I did not have that for this project. I chose an acceptance threshold of 0.15 after looking at the policy curve plots rather than picking the mathematically smallest possible expected loss or expected loss rate because that would accept little loans

The goal was to find a point where the tradeoff looked the most reasonable confirmed on the unseen test data:

- expected loss rate on accepted loans was still controlled
- total expected loss had not yet started increasing too aggressively
- acceptance rate still looked high enough to keep the portfolio useful
- moving materially higher than 0.15 appeared to increase risk more quickly
- moving materially lower than 0.15 appeared too restrictive by accepting less than 40% of loans relative to the extra benefit

So the threshold was selected as a practical middle ground rather than an extreme minimum-loss rule

## Interpretation

This threshold means:

- accept loans with predicted_prob_bad <= 0.15
- reject loans with predicted_prob_bad > 0.15

In this project, the threshold is being used as a first policy rule for expected-loss analysis.

## Expected loss framework

The policy analysis uses:

EL = PD × LGD × EAD

where:

- PD = predicted probability of default from the uncalibrated champion XGBoost model
- LGD = loss given default, assumed to be 0.40, but this is another metric I could not have a real value for within my project so this is just a constant which just scales
- EAD = exposure at default, proxied by loan_amnt


## Why the uncalibrated model was used

Both sigmoid and isotonic calibration were tested on a calibration slice

Although calibration improved fit on the calibration slice itself, neither method improved the final out-of-time test metrics relative to the uncalibrated XGBoost model. specifically neither improved the final test log loss or Brier score

Because of that, the uncalibrated XGBoost was retained as the champion model for policy analysis.

## Terminal check used to pull the acceptance rate for threshold 0.15 which is 0.4326

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("reports/tables/modeling_v1/policy_sweep_v1.csv")

row = df[(df["dataset"] == "test") & (df["threshold"].round(2) == 0.15)]

print(row[["dataset", "threshold", "acceptance_rate"]].to_string(index=False))
PY