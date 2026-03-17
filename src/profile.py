import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def bytes_to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

    
# Read loan data and put into datetime format
def read_data(input_path: str, date_col: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path, parse_dates=[date_col])
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")

    return df

# Compute shape, memory, duplicates, date range, and default rate 
def compute_overview(
    df: pd.DataFrame, date_col: str, target_col: str, default_value: str
) -> pd.DataFrame:
    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = bytes_to_mb(memory_bytes)

    dup_rows = int(df.duplicated().sum())

    if date_col in df.columns:
        date_min = df[date_col].min()
        date_max = df[date_col].max()
    else:
        date_min = None
        date_max = None

    if target_col in df.columns:
        y = (df[target_col] == default_value).astype(int)
        event_rate = float(y.mean())
    else:
        event_rate = None
    # Put all metrics into 2 col table
    overview_rows = [
        {"metric": "rows", "value": n_rows},
        {"metric": "columns", "value": n_cols},
        {"metric": "memory_mb", "value": memory_mb},
        {"metric": "duplicate_rows", "value": dup_rows},
        {"metric": "date_min", "value": str(date_min) if date_min is not None else ""},
        {"metric": "date_max", "value": str(date_max) if date_max is not None else ""},
        {"metric": "event_rate", "value": event_rate if event_rate is not None else ""},
        {"metric": "event_definition", "value": f"{target_col} == '{default_value}'"},
    ]

    return pd.DataFrame(overview_rows)

# Compute % missing per col
def compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing_pct = df.isna().mean() * 100
    out = (
        missing_pct.sort_values(ascending=False)
        .rename("pct_missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    out["pct_missing"] = out["pct_missing"].round(2)
    return out

# Compute unique values per categorical col
def compute_categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    rows = []
    for col in cat_cols:
        rows.append({"column": col, "n_unique": int(df[col].nunique(dropna=True))})
    out = pd.DataFrame(rows).sort_values("n_unique", ascending=False)
    return out

# Compute counts + percents for categorical col
def compute_value_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:

    counts = (
        df[col]
        .astype("string")
        .fillna("MISSING")
        .value_counts(dropna=False)
        .rename_axis(col)
        .reset_index(name="count")
    )

    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(4)
    return counts

# Plot top N missingness cols
def plot_missingness(missingness_df: pd.DataFrame, top_n: int, out_path: str) -> None:
    top = missingness_df.head(top_n).copy()
    top = top.sort_values("pct_missing", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top["column"], top["pct_missing"])
    plt.xlabel("Percent missing")
    plt.title(f"Top {top_n} columns by missingness")
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# (NEW) Plot missingness for a specific band to get better insight
def plot_missingness_band(
    missingness_df: pd.DataFrame,
    min_pct: float,
    max_pct: float,
    out_path: str,
    max_bars: int = 60,
) -> None:
    # Filter
    band = missingness_df[
        (missingness_df["pct_missing"] >= min_pct)
        & (missingness_df["pct_missing"] <= max_pct)
    ].copy()

    if band.empty:
        print(f"No columns found with missingness between {min_pct}% and {max_pct}%.")
        return

    # Sort
    band = band.sort_values("pct_missing", ascending=False)

    band = band.head(max_bars)

    plt.figure(figsize=(10, 8))
    plt.barh(band["column"], band["pct_missing"])
    plt.xlabel("Percent missing")
    plt.title(f"Columns with {min_pct}% to {max_pct}% missing (top {len(band)})")
    plt.gca().invert_yaxis()  # highest missingness at top
    plt.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_path = cfg["input_path"]
    date_col = cfg["date_col"]
    target_col = cfg["target_col"]
    default_value = cfg["default_value"]
    out_tables_dir = cfg["out_tables_dir"]
    out_figures_dir = cfg["out_figures_dir"]
    top_n = int(cfg.get("missingness_top_n", 30))

    ensure_dir(out_tables_dir)
    ensure_dir(out_figures_dir)

    df = read_data(input_path=input_path, date_col=date_col)

    overview = compute_overview(
        df=df, date_col=date_col, target_col=target_col, default_value=default_value
    )
    missingness = compute_missingness(df)
    cardinality = compute_categorical_cardinality(df)

    overview_path = str(Path(out_tables_dir) / "data_profile.csv")
    missingness_path = str(Path(out_tables_dir) / "missingness.csv")
    cardinality_path = str(Path(out_tables_dir) / "categorical_cardinality.csv")

    overview.to_csv(overview_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    cardinality.to_csv(cardinality_path, index=False)

    # Export target column value counts
    if target_col in df.columns:
        status_counts = compute_value_counts(df, target_col)
        status_counts_path = str(Path(out_tables_dir) / f"{target_col}_counts.csv")
        status_counts.to_csv(status_counts_path, index=False)

    fig_path = str(Path(out_figures_dir) / "missingness_top30.png")
    plot_missingness(missingness_df=missingness, top_n=top_n, out_path=fig_path)

    # more useful missingness plot that excludes the ~99-100% junk columns
    band_fig_path = str(Path(out_figures_dir) / "missingness_band_50_97.png")
    plot_missingness_band(
    missingness_df=missingness,
    min_pct=50.0,
    max_pct=97.0,
    out_path=band_fig_path,
    max_bars=60,
    )

    print("Wrote tables:")
    print(" -", overview_path)
    print(" -", missingness_path)
    print(" -", cardinality_path)
    print(" -", band_fig_path)

    # Print loan_status distribution table
    if target_col in df.columns:
        print(" -", status_counts_path)

    print("Wrote figure:")
    print(" -", fig_path)


if __name__ == "__main__":
    main()