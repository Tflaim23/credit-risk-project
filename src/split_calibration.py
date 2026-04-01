#Need to calibrate model so I will do a similar forward in time split as the original data just on the train data
#I thought this script would be more different but it is really just copy and paste of split.py with different paths and names for the dfs and a few other small changes but nothing much
#I had never learned about calibrating a model but knew I had to do it for this project
#I understood the concept of a forward in time split for train/test but for some reason did not realize it was essentially the exact same for a calibration split just using the train data rather than the full data
# I could have done this script in a few minutes just swapping names in hindsight but it was a good learning experience to realize first hand how the processes are essentially the same I think
import argparse
from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


#copy and paste from split.py 

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
    subtrain_output_path = cfg["subtrain_output_path"]
    calibration_output_path = cfg["calibration_output_path"]
    manifest_output_path = cfg["manifest_output_path"]
    #same as usual get the cols from yaml config but .get() can use a default as the second arg
    issue_date_col = cfg.get("issue_date_col", "issue_d")
    target_col = cfg.get("target_col", "y")
    calibration_fraction = float(cfg.get("calibration_fraction", 0.20))

    df = pd.read_parquet(input_path).copy()

    # Basically everything below is just copy and paste from split.py with few changes (like data to df) until the next note that says something is needed
    # I thought it would be more different but coming back to this, after this point basically nothing is new this script is super similar to split.py just with different paths and names but it worked fine so I guess that is fine

     #Turn issue_d into real dates with earlier function
    df[issue_date_col] = parse_issue_dates(df[issue_date_col])

    missing_issue_dates = int(df[issue_date_col].isna().sum())
    if missing_issue_dates > 0:
        raise ValueError(
            f"{issue_date_col} has {missing_issue_dates} missing dates."
        )


    # get unique issue dates in time order
    unique_dates = (
        pd.Series(df[issue_date_col].dropna().unique())
        .sort_values()
        .reset_index(drop=True)
    )

    n_unique_dates = len(unique_dates)
    if n_unique_dates < 2:
        raise ValueError("All dates are the same, formatting issue somewhere")

   # Reserve the last 20% (or whatever number is input for test_fraction) of time for calibration
    n_calibration_dates = int(round(n_unique_dates * calibration_fraction))

    split_date = unique_dates.iloc[n_unique_dates - n_calibration_dates]

    subtrain_df = df[df[issue_date_col] < split_date].copy()
    calibration_df = df[df[issue_date_col] >= split_date].copy()

    if len(subtrain_df) == 0 or len(calibration_df) == 0:
        raise ValueError("Train or test (or both) is empty something failed with the split")

    ensure_parent_dir(subtrain_output_path)
    ensure_parent_dir(calibration_output_path)
    ensure_parent_dir(manifest_output_path)

    subtrain_df.to_parquet(subtrain_output_path, index=False)
    calibration_df.to_parquet(calibration_output_path, index=False)

    #I still copy and pasted this from split.py but has to change a lot of the values obviously
    manifest = pd.DataFrame(
        [
            {
                "split": "subtrain",
                "rows": len(subtrain_df),
                "bad_rate": round(subtrain_df[target_col].mean(), 5),
                "min_issue_d": subtrain_df[issue_date_col].min().date(),
                "max_issue_d": subtrain_df[issue_date_col].max().date(),
                "n_unique_issue_dates": subtrain_df[issue_date_col].nunique(),
            },
            {
                "split": "calibration",
                "rows": len(calibration_df),
                "bad_rate": round(calibration_df[target_col].mean(), 5),
                "min_issue_d": calibration_df[issue_date_col].min().date(),
                "max_issue_d": calibration_df[issue_date_col].max().date(),
                "n_unique_issue_dates": calibration_df[issue_date_col].nunique(),
            },
        ]
    )

    manifest.to_csv(manifest_output_path, index=False)

    print("input:", input_path)
    print("rows:", len(df))
    print("Unique issue dates:", n_unique_dates)
    print("calibration fraction:", calibration_fraction)
    print("Split date:", split_date.date())
    print("Subtrain rows:", len(subtrain_df))
    print("Calibration rows:", len(calibration_df))
    print("wrote:")
    print(" -", subtrain_output_path)
    print(" -", calibration_output_path)
    print(" -", manifest_output_path)


if __name__ == "__main__":
    main()