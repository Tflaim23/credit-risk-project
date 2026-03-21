# Make a list and remove y and issue d because those cols have nothing to do with predicting they were just needed eaarlier 
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    output_path = cfg["output_path"]
    remove_columns = set(cfg.get("remove_columns", []))

    keep_df = pd.read_csv(input_path)

    # Only need the col names from keep file
    feature_names = keep_df["column"].copy()

    # Remove y and issue d (in yaml)
    feature_names = feature_names[~feature_names.isin(remove_columns)]

    # sort alphabetically
    feature_names = feature_names.sort_values().reset_index(drop=True)

    feature_df = pd.DataFrame({"feature": feature_names})

    ensure_output_dir(output_path)
    feature_df.to_csv(output_path, index=False)

    print("Input:", input_path)
    print("Rows in keep file:", len(keep_df))
    print("Removed columns:", sorted(remove_columns))
    print("Final feature count:", len(feature_df))
    print("Wrote:")
    print(" -", output_path)


if __name__ == "__main__":
    main()