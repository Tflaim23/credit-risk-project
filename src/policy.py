#Goal of the script is to evaluate acceptance threshold policy based on expected loss formula
#Use subtrain to choose the threshold then compare it on the untouched test
#I will make a table with all of the thresholds tested
# The goal will be to find the threshold with the highest acceptance rate where expected loss is still acceptable
#From what I have looked at there is no firm rule but generally when dropping acceptance stops significntly decreasing expected loss is when yous top

import argparse
from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


#List of thresholds based on my config
def make_threshold_list(start: float, stop: float, step: float) -> list[float]:
    thresholds = []
    current = start

    while current <= stop + 0.00001:
        thresholds.append(round(current, 2))
        current += step

    return thresholds

#accept a loan if predicted_prob_bad <= threshold
#This function gets me all the metrics I need for one threshold to pick the most ideal threshold
def evaluate_policy_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    lgd: float,
    dataset_name: str,
) -> dict:
    #df that only shows accepted loans 
    accepted = df.loc[df["predicted_prob_bad"] <= threshold].copy()

    total_loans = len(df)
    accepted_loans = len(accepted)

    acceptance_rate = accepted_loans / total_loans

    #no accepted loans with an extremely low threshold
    if accepted_loans == 0:
        return {
        "dataset": dataset_name,
        "threshold": threshold,
        "accepted_loans": 0,
        "note": "no loans accepted"
    }
    # add up loan amt of all accepted loans
    total_ead_accepted = float(accepted["ead"].sum())

    # EL = PD×LGD×EAD for each row because it is vectorized
    expected_loss_accepted = float(
        (accepted["predicted_prob_bad"] * lgd * accepted["ead"]).sum()
    )

    #Gets the rate of excepted loss to total loan amt accepted to get a better picture and make it so you can compare fairly across thresholds with different acceptance rates 
    expected_loss_rate_accepted = expected_loss_accepted / total_ead_accepted
    

    # Same things but gets the actual value using true 1/0 default values rather than the models predicted probabilities
    actual_loss_accepted = float(
        (accepted["actual_bad"] * lgd * accepted["ead"]).sum()
    )

    actual_loss_rate_accepted = actual_loss_accepted / total_ead_accepted

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "total_loans": total_loans,
        "accepted_loans": accepted_loans,
        "acceptance_rate": round(float(acceptance_rate), 4),
        "observed_bad_rate_accepted": round(float(accepted["actual_bad"].mean()), 4),
        "avg_predicted_pd_accepted": round(float(accepted["predicted_prob_bad"].mean()), 4),
        "total_ead_accepted": round(total_ead_accepted, 2),
        "expected_loss_accepted": round(expected_loss_accepted, 2),
        "expected_loss_rate_accepted": round(float(expected_loss_rate_accepted), 4),
        "actual_loss_accepted": round(actual_loss_accepted, 2),
        "actual_loss_rate_accepted": round(float(actual_loss_rate_accepted), 4),
    }

#takes the df and runs the last function for every threshold in the list
def build_policy_table(
    df: pd.DataFrame,
    thresholds: list[float],
    lgd: float,
    dataset_name: str,
) -> pd.DataFrame:
    
    rows = []

    #loop through and call the metric function for all thresholds
    for threshold in thresholds:
        row = evaluate_policy_at_threshold(
            df=df,
            threshold=threshold,
            lgd=lgd,
            dataset_name=dataset_name,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    subtrain_input_path = cfg["subtrain_input_path"]
    test_input_path = cfg["test_input_path"]

    sweep_output_path = cfg["sweep_output_path"]

    lgd = float(cfg.get("lgd", 0.4))
    threshold_min = float(cfg.get("threshold_min", 0.01))
    threshold_max = float(cfg.get("threshold_max", 0.99))
    threshold_step = float(cfg.get("threshold_step", 0.01))

    subtrain_df = pd.read_csv(subtrain_input_path)
    test_df = pd.read_csv(test_input_path)

    thresholds = make_threshold_list(threshold_min, threshold_max, threshold_step)

    subtrain_table = build_policy_table(
        df=subtrain_df,
        thresholds=thresholds,
        lgd=lgd,
        dataset_name="subtrain",
    )

    test_table = build_policy_table(
        df=test_df,
        thresholds=thresholds,
        lgd=lgd,
        dataset_name="test",
    )

    sweep_df = pd.concat([subtrain_table, test_table], ignore_index=True)

    ensure_parent_dir(sweep_output_path)

    sweep_df.to_csv(sweep_output_path, index=False)

    print("LGD:", lgd)
    print("Threshold count:", len(thresholds))
    print()
    print("wrote:")
    print(" -", sweep_output_path)


if __name__ == "__main__":
    main()