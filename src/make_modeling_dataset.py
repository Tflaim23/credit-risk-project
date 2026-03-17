import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Pulls all the settings from yaml
    input_path = cfg["input_path"]
    date_col = cfg["date_col"]
    status_col = cfg["status_col"]
    good_statuses = set(cfg["good_statuses"])
    bad_statuses = set(cfg["bad_statuses"])
    out_path = cfg["out_processed_path"]

    df = pd.read_parquet(input_path)

    # Date is datetime
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Filter to only reliable outcomes
    keep = df[status_col].isin(good_statuses.union(bad_statuses))
    df = df.loc[keep].copy()

    # Create target label: 1=bad, 0=good
    df["y"] = df[status_col].isin(bad_statuses).astype(int)

    ensure_parent_dir(out_path)
    df.to_parquet(out_path, index=False)

    print(f"Input:  {input_path}")
    print(f"Output: {out_path}")
    print(f"Kept rows: {len(df):,}")
    print(f"Bad rate:", round(df["y"].mean(), 6))


if __name__ == "__main__":
    main()