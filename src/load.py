import argparse
from pathlib import Path

import pandas as pd
import yaml

# Pull settings as python dict from YAML
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    #Parser to read what yaml config file to use when running the script. This allows it to have different config 
    # run scripts by doing something like:
    # First line: source .venv/bin/activate
    # Second line: python src/load.py --config configs/load_config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    output_path = cfg["output_path"]
    date_cols = cfg.get("date_cols", [])

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=date_cols)

    # write to Parquet
    ensure_parent_dir(output_path)
    df.to_parquet(output_path, index=False)

    print(f"Read:  {input_path} (rows={len(df):,}, cols={df.shape[1]})")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()