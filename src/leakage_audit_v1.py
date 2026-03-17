# Drops cols based on predetirmined must_drop, missingness or cardinality
# Most importanntly drops leakage cols which have information that ony came in after issue d

import argparse
import re
from pathlib import Path
import pandas as pd
import yaml
#easier to call
from pandas.api.types import is_object_dtype, is_string_dtype


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dictionary(dictionary_csv: str) -> pd.DataFrame:
    p = Path(dictionary_csv)
    #If lendingclub dictionary isn't there it won't error the script
    if not p.exists():
        return pd.DataFrame(columns=["field", "description"])

    d = pd.read_csv(p)
    d["field"] = d["field"].astype("string").str.strip()
    d["description"] = d["description"].astype("string").str.strip()
    return d


def is_text_column(series: pd.Series) -> bool:
    return is_object_dtype(series) or is_string_dtype(series)


def matches_any_pattern(column_name: str, patterns: list[str]) -> bool:
    column_name = column_name.lower()
# Search col names for anything in leakage patterns list in yaml
    for pattern in patterns:
        if re.search(pattern, column_name):
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    dictionary_csv = cfg.get("dictionary_csv", "")
    output_dir = cfg["output_dir"]

    missing_threshold_pct = float(cfg.get("missing_threshold_pct", 95.0))
    high_cardinality_threshold = int(cfg.get("high_cardinality_threshold", 500))

    must_keep = set(cfg.get("must_keep", []))
    must_drop = set(cfg.get("must_drop", []))
    leakage_patterns = cfg.get("leakage_name_patterns", [])

    ensure_dir(output_dir)

    # Read the modeling dataset.
    df = pd.read_parquet(input_path)

    # This errored for me many times so I had to do it this way
    # One summary row per column
    # For each column store:
    # dtype
    # % missing
    # text (T/F)
    # # of unique values if T
    rows = []

    for col in df.columns:
        series = df[col]

        dtype_name = str(series.dtype)
        pct_missing = round(series.isna().mean() * 100, 2)

        text_flag = is_text_column(series)

        if text_flag:
            n_unique = int(series.nunique(dropna=True))
        else:
            n_unique = 0

        rows.append(
            {
                "column": col,
                "dtype": dtype_name,
                "pct_missing": pct_missing,
                "is_text_column": text_flag,
                "n_unique": n_unique,
            }
        )

    stats = pd.DataFrame(rows)

    # Add descriptions from the dictionary to stats
    dd = load_dictionary(dictionary_csv)

    if len(dd) > 0:
        stats = stats.merge(dd, left_on="column", right_on="field", how="left")
        stats = stats.drop(columns=["field"])
    else:
        stats["description"] = ""

    # Decide why each column should be dropped
    drop_reasons = []

    for _, row in stats.iterrows():
        reasons = []

        col = row["column"]
        pct_missing = float(row["pct_missing"])
        is_text = bool(row["is_text_column"])
        n_unique = int(row["n_unique"])

        if col in must_drop:
            reasons.append("MUST_DROP")

        if pct_missing >= missing_threshold_pct:
            reasons.append("HIGH_MISSING")

        if is_text and n_unique >= high_cardinality_threshold:
            reasons.append("HIGH_CARDINALITY")

        if matches_any_pattern(col, leakage_patterns):
            reasons.append("LEAKAGE_NAME_PATTERN")

        # Final override
        if col in must_keep:
            reasons = []

        drop_reasons.append("|".join(reasons))

    stats["drop_reasons"] = drop_reasons

    drop_df = stats[stats["drop_reasons"] != ""].copy()
    keep_df = stats[stats["drop_reasons"] == ""].copy()

    drop_df = drop_df.sort_values(["pct_missing", "n_unique"], ascending=[False, False])
    keep_df = keep_df.sort_values("column")

    drop_path = Path(output_dir) / "drop_columns_v1.csv"
    keep_path = Path(output_dir) / "keep_columns_v1.csv"

    drop_df.to_csv(drop_path, index=False)
    keep_df.to_csv(keep_path, index=False)

    print("input:", input_path)
    print("total columns:", df.shape[1])
    print("drop columns:", len(drop_df))
    print("keep columns:", len(keep_df))
    print("wrote:")
    print(" -", drop_path)
    print(" -", keep_path)


if __name__ == "__main__":
    main()