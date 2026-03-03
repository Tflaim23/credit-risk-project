import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    # Reminders to add inouts and ouputs if forgotten and stores the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    # Read the LoanStats sheet
    df = pd.read_excel(args.input_xlsx, sheet_name="LoanStats")

    # Standardize column names
    df = df.rename(columns={"LoanStatNew": "field", "Description": "description"})

    # Clean up
    df["field"] = df["field"].astype("string").str.strip()
    df["description"] = df["description"].astype("string").str.strip()
    df = df.dropna(subset=["field"])
    df = df[df["field"] != ""]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"CSV: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
