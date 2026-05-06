# this script is stress testing how much expected loss on the accepted loan portfolio increases under stressed PD and LGD assumptions
# this is for understanding downside risk
# especially important given evidence that the model slightly underestimates default risk as seen in stability
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# stressing a portfolio which has already been accepted by the selected threshold
# the idea is to not change the loans in the portfolio only increasing assumtions
def evaluate_fixed_portfolio_scenario(
    accepted_df: pd.DataFrame,
    base_lgd: float,
    scenario_name: str,
    pd_multiplier: float,
    lgd_multiplier: float,
) -> dict:
    
    # This will be coming from main function where I create df from the config input like usual
    # Only the loans with pd <= threshold which get accepted
    stressed_df = accepted_df.copy()

    # pd and lgd multipliers applied
    # clip is for a pd series to keep probs under one and for the single variable just do min with 1.0
    stressed_df["stressed_pd"] = (stressed_df["predicted_prob_bad"] * pd_multiplier).clip(0, 1)
    stressed_lgd = min(base_lgd * lgd_multiplier, 1.0)

    total_loans = len(stressed_df)
    total_ead = float(stressed_df["ead"].sum())

    avg_base_pd = float(stressed_df["predicted_prob_bad"].mean())
    avg_stressed_pd = float(stressed_df["stressed_pd"].mean())

    expected_loss = float(
        (stressed_df["stressed_pd"] * stressed_lgd * stressed_df["ead"]).sum()
    )

    expected_loss_rate = expected_loss / total_ead

    actual_bad_rate = float(stressed_df["actual_bad"].mean())
    actual_loss = float((stressed_df["actual_bad"] * base_lgd * stressed_df["ead"]).sum())
    actual_loss_rate = actual_loss / total_ead

    # return everything in a dict
    return {
        "scenario": scenario_name,
        "pd_multiplier": pd_multiplier,
        "stressed_lgd": round(stressed_lgd, 4),
        "accepted_loans_fixed": total_loans,
        "total_ead_fixed": round(total_ead, 2),
        "avg_base_pd": round(avg_base_pd, 4),
        "avg_stressed_pd": round(avg_stressed_pd, 4),
        "actual_bad_rate_fixed_portfolio": round(actual_bad_rate, 4),
        "expected_loss_fixed_portfolio": round(expected_loss, 2),
        "expected_loss_rate_fixed_portfolio": round(expected_loss_rate, 4),
        "actual_loss_fixed_portfolio": round(actual_loss, 2),
        "actual_loss_rate_fixed_portfolio": round(actual_loss_rate, 4),
    }


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    output_table_path = cfg["output_table_path"]
    loss_plot_path = cfg["loss_plot_path"]
    loss_rate_plot_path = cfg["loss_rate_plot_path"]

    threshold = float(cfg.get("threshold", 0.15))
    base_lgd = float(cfg.get("base_lgd", 0.40))
    scenarios = cfg["scenarios"]

    df = pd.read_csv(input_path)

    # Freeze the originally accepted portfolio
    accepted_df = df[df["predicted_prob_bad"] <= threshold].copy()

    rows = []

    # Run it for all 4 scenarios
    for scenario in scenarios:
        row = evaluate_fixed_portfolio_scenario(
            accepted_df=accepted_df,
            base_lgd=base_lgd,
            scenario_name=scenario["name"],
            pd_multiplier=float(scenario["pd_multiplier"]),
            lgd_multiplier=float(scenario["lgd_multiplier"]),
        )
        rows.append(row)

    results = pd.DataFrame(rows)

    ensure_parent_dir(output_table_path)
    ensure_parent_dir(loss_plot_path)
    ensure_parent_dir(loss_rate_plot_path)

    results.to_csv(output_table_path, index=False)

    # total expected loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["scenario"], results["expected_loss_fixed_portfolio"], marker="o")
    plt.xlabel("Scenario")
    plt.ylabel("Expected Loss on Fixed Accepted Portfolio")
    plt.title("")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    # loss rate plot
    # compares stressed expected loss rate and actual-loss proxy rate
    plt.figure(figsize=(10, 6))
    plt.plot(results["scenario"], results["expected_loss_rate_fixed_portfolio"], marker="o", label="Expected Loss Rate")
    plt.plot(results["scenario"], results["actual_loss_rate_fixed_portfolio"], marker="o", label="Actual Loss Rate")
    plt.xlabel("Scenario")
    plt.ylabel("Loss Rate")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_rate_plot_path, dpi=150)
    plt.close()

    print("Threshold:", threshold)
    print("Base lgd:", base_lgd)
    print("# of accepted loans in fixed portfolio:", len(accepted_df))
    print()
    print(results.to_string(index=False))
    print()
    print("Wrote:")
    print(" -", output_table_path)
    print(" -", loss_plot_path)
    print(" -", loss_rate_plot_path)


if __name__ == "__main__":
    main()