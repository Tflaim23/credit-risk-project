#Sending earlier dates to training data and later to test
import argparse
from pathlib import Path

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

    input_path = cfg["input_path"]
    feature_set_path = cfg["feature_set_path"]

    train_output_path = cfg["train_output_path"]
    test_output_path = cfg["test_output_path"]
    manifest_output_path = cfg["manifest_output_path"]
    #add defaults to issue_d and y incase I forget to add them to yaml config
    issue_date_col = cfg.get("issue_date_col", "issue_d")
    target_col = cfg.get("target_col", "y")
    test_fraction = float(cfg.get("test_fraction"))

    df = pd.read_parquet(input_path)
    feature_df = pd.read_csv(feature_set_path)

    features = feature_df["feature"].tolist()

    needed_columns = features + [issue_date_col, target_col]

    missing_columns = [col for col in needed_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing cols: {missing_columns}")

    # Keep only the frozen feature set plus issue_d and y as long as it did not error out above
    data = df[needed_columns].copy()

    # Turn issue_d into real dates with earlier function
    data[issue_date_col] = parse_issue_dates(data[issue_date_col])

    missing_issue_dates = int(data[issue_date_col].isna().sum())
    if missing_issue_dates > 0:
        raise ValueError(
            f"{issue_date_col} has {missing_issue_dates} missing dates."
        )

    # get unique issue dates in time order
    unique_dates = (
        pd.Series(data[issue_date_col].dropna().unique())
        .sort_values()
        .reset_index(drop=True)
    )

    n_unique_dates = len(unique_dates)
    if n_unique_dates < 2:
        raise ValueError("All dates are the same, formatting issue somewhere")

    # Reserve the last 20% (or whatever number is input for test_fraction) of time for test
    n_test_dates = int(round(n_unique_dates * test_fraction))

    # Finds the cutoff
    split_date = unique_dates.iloc[n_unique_dates - n_test_dates]

    train_df = data[data[issue_date_col] < split_date].copy()
    test_df = data[data[issue_date_col] >= split_date].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Train or test (or both) is empty something failed with the split")

    ensure_parent_dir(train_output_path)
    ensure_parent_dir(test_output_path)
    ensure_parent_dir(manifest_output_path)

    train_df.to_parquet(train_output_path, index=False)
    test_df.to_parquet(test_output_path, index=False)
    
    #Came back to create this after trying to run the script initally.
    #essentially creating a manifest file with some basic info about the train and test splits to understand what is going on and help debug
    manifest = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "bad_rate": round(train_df[target_col].mean(), 2),
                "min_issue_d": train_df[issue_date_col].min().date(),
                "max_issue_d": train_df[issue_date_col].max().date(),
                "n_unique_issue_dates": train_df[issue_date_col].nunique(),
            },
            {
                "split": "test",
                "rows": len(test_df),
                "bad_rate": round(test_df[target_col].mean(), 2),
                "min_issue_d": test_df[issue_date_col].min().date(),
                "max_issue_d": test_df[issue_date_col].max().date(),
                "n_unique_issue_dates": test_df[issue_date_col].nunique(),
            },
        ]
    )

    manifest.to_csv(manifest_output_path, index=False)

#Had some errors for a long time so printed everything in depth to ensure it is working right
    print("Input:", input_path)
    print("Feature set:", feature_set_path)
    print("Rows:", len(data))
    print("Feature count:", len(features))
    print("Unique issue dates:", n_unique_dates)
    print("Test fraction:", test_fraction)
    print("Split date:", split_date.date())
    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Wrote:")
    print(" -", train_output_path)
    print(" -", test_output_path)
    print(" -", manifest_output_path)


if __name__ == "__main__":
    main()