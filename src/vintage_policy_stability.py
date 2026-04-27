#Stability is to check whether chosen policy threshold behaves consistently over time on the test
#Basically just making sure risk stays stable in different time periods
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



# Try parsing issue dates with "Mon-YYYY" format which should work
# falls back to more flexible parsing if needed
#similar idea to creating the parent folders if they don't exist in that it is unlikely to cause a problem but just in case
def parse_issue_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")

    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")

    return parsed



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_test_path = cfg["raw_test_path"]
    policy_input_path = cfg["policy_input_path"]

    issue_date_col = cfg.get("issue_date_col", "issue_d")
    threshold = float(cfg.get("threshold", 0.15))
    vintage_freq = cfg.get("vintage_freq", "Q")

    output_table_path = cfg["output_table_path"]
    acceptance_plot_path = cfg["acceptance_plot_path"]
    risk_plot_path = cfg["risk_plot_path"]

    raw_test = pd.read_parquet(raw_test_path)
    policy_df = pd.read_csv(policy_input_path)

    df = policy_df.copy()
    df[issue_date_col] = parse_issue_dates(raw_test[issue_date_col])

    max_issue_date_exclusive = cfg.get("max_issue_date_exclusive")

    # trimming of recent immature vintages
    cutoff = pd.to_datetime(max_issue_date_exclusive)
    df = df[df[issue_date_col] < cutoff].copy()

    # policy rule: accept if predicted bad probability is at or below the threshold
    df["accepted"] = (df["predicted_prob_bad"] <= threshold).astype(int)

    if vintage_freq == "Q":
        df["vintage"] = df[issue_date_col].dt.to_period("Q").dt.start_time
    elif vintage_freq == "Y":
        df["vintage"] = df[issue_date_col].dt.to_period("Y").dt.start_time
    else:
        raise ValueError("make config freq yearly or quarterly")

    rows = []

    # loop through each vintage
    #For each time period take all loans from that period and compute how the policy behaves on just those loans
    #vintage is the label for each time period and temp is the group of loans from that vintage
    for vintage, temp in df.groupby("vintage"):
        # filters the accepted loans in that vintage
        accepted = temp[temp["accepted"] == 1].copy()

        total_loans = len(temp)
        accepted_loans = len(accepted)

        acceptance_rate = accepted_loans / total_loans
        overall_bad_rate = temp["actual_bad"].mean()

        if accepted_loans > 0:
            accepted_bad_rate = accepted["actual_bad"].mean()
            avg_predicted_pd_accepted = accepted["predicted_prob_bad"].mean()
            total_ead_accepted = accepted["ead"].sum()
        else:
            accepted_bad_rate = float("nan")
            avg_predicted_pd_accepted = float("nan")
            total_ead_accepted = 0.0

        rows.append(
            {
                "vintage": vintage,
                "threshold": threshold,
                "total_loans": total_loans,
                "accepted_loans": accepted_loans,
                "acceptance_rate": round(float(acceptance_rate), 4),
                "overall_bad_rate": round(float(overall_bad_rate), 4),
                "accepted_bad_rate": round(float(accepted_bad_rate), 4) if pd.notna(accepted_bad_rate) else float("nan"),
                "avg_predicted_pd_accepted": round(float(avg_predicted_pd_accepted), 4) if pd.notna(avg_predicted_pd_accepted) else float("nan"),
                "total_ead_accepted": round(float(total_ead_accepted), 2),
            }
        )
    #oldest first for the summary table 
    summary_df = pd.DataFrame(rows).sort_values("vintage").reset_index(drop=True)

    ensure_parent_dir(output_table_path)
    ensure_parent_dir(acceptance_plot_path)
    ensure_parent_dir(risk_plot_path)

    summary_df.to_csv(output_table_path, index=False)

    # acceptance rate by vintage
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["vintage"], summary_df["acceptance_rate"], marker="o")
    plt.xlabel("Vintage")
    plt.ylabel("Acceptance Rate")
    plt.title("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(acceptance_plot_path, dpi=150)
    plt.close()

    # accepted bad rate vs avg predicted PD among accepted loans
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["vintage"], summary_df["accepted_bad_rate"], marker="o", label="Accepted Bad Rate")
    plt.plot(summary_df["vintage"], summary_df["avg_predicted_pd_accepted"], marker="o", label="Avg Predicted PD Accepted")
    plt.xlabel("Vintage")
    plt.ylabel("Rate")
    plt.title("")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(risk_plot_path, dpi=150)
    plt.close()

    print("Threshold:", threshold)
    print("Vintage frequency:", vintage_freq)
    print("Max issue date exclusive:", max_issue_date_exclusive)
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Wrote:")
    print(" -", output_table_path)
    print(" -", acceptance_plot_path)
    print(" -", risk_plot_path)


if __name__ == "__main__":
    main()