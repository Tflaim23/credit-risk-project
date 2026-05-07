# This monte carlo simulation will randomly stress pd and lgd thousands of times to see the distribution of possible expected losses
# This is important because the fixed stress test gave a few examples
# this simulation is more comprehensive because it provides percentiles like median, 90th, and 95th percentile loss

# For each of the 4 scenarios I run 2,000 simulations and in each of those randomly draw a PD multiplier and LGD multiplier
# Recalculate the EL for each sim and save everything
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
#for rng
import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# Draw one random multiplier from a norm dist
# clip it at zero so we never get a negative multiplier
def draw_multiplier(mean: float, std: float, rng: np.random.Generator) -> float:
    value = rng.normal(loc=mean, scale=std)
    return max(float(value), 0.0)

# calculates expected loss for one simulation
def evaluate_one_simulation(
    accepted_df: pd.DataFrame,
    base_lgd: float,
    pd_multiplier: float,
    lgd_multiplier: float,
) -> dict:

    # I dont't copy the df in this one because it's going to run thousand of sims
    stressed_pd = (accepted_df["predicted_prob_bad"] * pd_multiplier).clip(0, 1)
    stressed_lgd = min(base_lgd * lgd_multiplier, 1.0)

    total_ead = float(accepted_df["ead"].sum())

    expected_loss = float((stressed_pd * stressed_lgd * accepted_df["ead"]).sum())
    expected_loss_rate = expected_loss / total_ead

    avg_stressed_pd = float(stressed_pd.mean())

    return {
        "pd_multiplier": round(pd_multiplier, 4),
        "lgd_multiplier": round(lgd_multiplier, 4),
        "stressed_lgd": round(stressed_lgd, 4),
        "avg_stressed_pd": round(avg_stressed_pd, 4),
        "expected_loss": round(expected_loss, 2),
        "expected_loss_rate": round(expected_loss_rate, 4),
    }

# summarize the 2,000 sims for all 4 scenarios one row for each
# quantile from pd
def summarize_scenario(sim_df: pd.DataFrame, scenario_name: str) -> dict:
    el = sim_df["expected_loss"]
    el_rate = sim_df["expected_loss_rate"]

    return {
        "scenario": scenario_name,
        "n_sims": int(len(sim_df)),
        "mean_expected_loss": round(float(el.mean()), 2),
        "std_expected_loss": round(float(el.std(ddof=1)), 2),
        "p05_expected_loss": round(float(el.quantile(0.05)), 2),
        "p50_expected_loss": round(float(el.quantile(0.50)), 2),
        "p90_expected_loss": round(float(el.quantile(0.90)), 2),
        "p95_expected_loss": round(float(el.quantile(0.95)), 2),
        "max_expected_loss": round(float(el.max()), 2),
        "mean_expected_loss_rate": round(float(el_rate.mean()), 4),
        "p50_expected_loss_rate": round(float(el_rate.quantile(0.50)), 4),
        "p95_expected_loss_rate": round(float(el_rate.quantile(0.95)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    simulations_output_path = cfg["simulations_output_path"]
    summary_output_path = cfg["summary_output_path"]
    histogram_output_path = cfg["histogram_output_path"]

    threshold = float(cfg.get("threshold", 0.15))
    base_lgd = float(cfg.get("base_lgd", 0.40))
    random_state = int(cfg.get("random_state", 42))
    scenarios = cfg["scenarios"]

    df = pd.read_csv(input_path)

    # Freeze the originally accepted portfolio
    accepted_df = df[df["predicted_prob_bad"] <= threshold].copy()

    rng = np.random.default_rng(random_state)

    all_sim_rows = []
    summary_rows = []

    # Loops through each scenario from config and gets all my values
    for scenario in scenarios:
        scenario_name = scenario["name"]
        n_sims = int(scenario["n_sims"])

        pd_mean = float(scenario["pd_multiplier_mean"])
        pd_std = float(scenario["pd_multiplier_std"])
        lgd_mean = float(scenario["lgd_multiplier_mean"])
        lgd_std = float(scenario["lgd_multiplier_std"])

        scenario_rows = []

        #runs sims from 1 to n which is 2000, +1 because it stops before the endpoint
        for sim_id in range(1, n_sims + 1):
            # Call rng function for pd and lgd
            pd_multiplier = draw_multiplier(pd_mean, pd_std, rng)
            lgd_multiplier = draw_multiplier(lgd_mean, lgd_std, rng)

            # Call the eval function
            sim_result = evaluate_one_simulation(
                accepted_df=accepted_df,
                base_lgd=base_lgd,
                pd_multiplier=pd_multiplier,
                lgd_multiplier=lgd_multiplier,
            )

            sim_result["scenario"] = scenario_name
            sim_result["sim_id"] = sim_id

            scenario_rows.append(sim_result)
            # Adds simulation number
            all_sim_rows.append(sim_result)

        scenario_df = pd.DataFrame(scenario_rows)
        summary_rows.append(summarize_scenario(scenario_df, scenario_name))

    # Full simulation level output, one row per simulation
    simulations_df = pd.DataFrame(all_sim_rows)
    # one row per scenario
    summary_df = pd.DataFrame(summary_rows)

    ensure_parent_dir(simulations_output_path)
    ensure_parent_dir(summary_output_path)
    ensure_parent_dir(histogram_output_path)

    simulations_df.to_csv(simulations_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    plt.figure(figsize=(10, 6))

    for scenario_name in simulations_df["scenario"].unique():
        temp = simulations_df[simulations_df["scenario"] == scenario_name]
        #alpha is for transparency so you can see overlapping histograms
        # bins is just # of bars essentially, each scenario gets 30 bins with sizes based on their range
        plt.hist(temp["expected_loss_rate"], bins=30, alpha=0.5, label=scenario_name)

    plt.xlabel("Expected Loss Rate")
    plt.ylabel("Frequency")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(histogram_output_path, dpi=150)
    plt.close()

    print("Threshold:", threshold)
    print("Base LGD:", base_lgd)
    print("Accepted loans in fixed portfolio:", len(accepted_df))
    print()
    print("Summary")
    print(summary_df.to_string(index=False))
    print()
    print("Wrote:")
    print(" -", simulations_output_path)
    print(" -", summary_output_path)
    print(" -", histogram_output_path)


if __name__ == "__main__":
    main()