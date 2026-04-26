#creates plots showing how expected loss changes as the acceptance threshold moves
# Makes it so I can visually see the tradeoff point where accepting more loans stops being worth the extra risk

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    input_path = "reports/tables/modeling_v1/policy_sweep_v1.csv"
    output_dir = Path("reports/figures/modeling_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_loss_rate_plot = output_dir / "policy_expected_loss_rate_by_threshold_v1.png"
    expected_loss_plot = output_dir / "policy_expected_loss_by_threshold_v1.png"

    df = pd.read_csv(input_path)

    #separates subtrain and test and orders them by threshold
    #chose the policy on subtrain then apply it to test
    subtrain = df[df["dataset"] == "subtrain"].copy().sort_values("threshold")
    test = df[df["dataset"] == "test"].copy().sort_values("threshold")

    print("input rows:", len(df))
    print("subtrain rows:", len(subtrain))
    print("test rows:", len(test))
    print()
    print("Subtrain range:", subtrain["threshold"].min(), "-", subtrain["threshold"].max())
    print("Test range:", test["threshold"].min(), "-", test["threshold"].max())
   
    # expected loss rate
    plt.figure(figsize=(10, 6))
    plt.plot(subtrain["threshold"], subtrain["expected_loss_rate_accepted"], marker="o", label="subtrain")
    plt.plot(test["threshold"], test["expected_loss_rate_accepted"], marker="o", label="test")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Loss Rate")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(expected_loss_rate_plot, dpi=150)
    plt.close()

    # Total expected loss 
    plt.figure(figsize=(10, 6))
    plt.plot(subtrain["threshold"], subtrain["expected_loss_accepted"], marker="o", label="subtrain")
    plt.plot(test["threshold"], test["expected_loss_accepted"], marker="o", label="test")
    plt.xlabel("Threshold")
    plt.ylabel("Total Expected Loss")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(expected_loss_plot, dpi=150)
    plt.close()

    print("Wrote:")
    print(" -", expected_loss_rate_plot)
    print(" -", expected_loss_plot)


if __name__ == "__main__":
    main()