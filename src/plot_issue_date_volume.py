#I want to check the distribution of loan dates before moving on because the calibration forward in time split was strange
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# no config file since I don't think I will need to reuse this ever
input_path = "data/processed/lendingclub_modeling_v1.parquet"
issue_date_col = "issue_d"
output_table_path = "reports/tables/modeling_v1/issue_month_counts_v1.csv"
output_figure_path = "reports/figures/modeling_v1/issue_month_counts_v1.png"
#date that split test and train
train_test_split_date = "2016-09-01"
#date that splits subtrain and calibration (from original train)
calibration_split_date = "2016-01-01"

def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

#same block from split.py to convert the issue_d column from text into real datetime objects
def parse_issue_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed

df = pd.read_parquet(input_path)

if issue_date_col not in df.columns:
    raise ValueError("Probably wrong input path")

df = df[[issue_date_col]].copy()
df[issue_date_col] = parse_issue_dates(df[issue_date_col])

missing_dates = int(df[issue_date_col].isna().sum())
if missing_dates > 0:
    raise ValueError("something wrong with the dates")

# drop the day then convert each issue date to the first day of its month to be able to group them
df["issue_month"] = df[issue_date_col].dt.to_period("M").dt.to_timestamp()

#Group by month, count rows in each group, two col table with issue_month and loan_count, sorted by lowest month first
monthly_counts = (
    df.groupby("issue_month")
    .size()
    .reset_index(name="loan_count")
    .sort_values("issue_month")
    .reset_index(drop=True)
)

ensure_parent_dir(output_table_path)
ensure_parent_dir(output_figure_path)

monthly_counts.to_csv(output_table_path, index=False)

plt.figure(figsize=(12, 6))
#x and y inputs
plt.plot(monthly_counts["issue_month"], monthly_counts["loan_count"])
plt.title("")
plt.xlabel("Month")
plt.ylabel("# of loans")
plt.xticks(rotation=45)

#add line on the graph at the train test split
#
plt.axvline(pd.to_datetime(train_test_split_date), linestyle="--")
plt.text(
    pd.to_datetime(train_test_split_date),
    #Put the line low down so I can read it
    monthly_counts["loan_count"].max() * 0.3,
    "Train/Test Split",
    rotation=90,
    va="top",
    )

#add line on the grsaph at the calibration split
plt.axvline(pd.to_datetime(calibration_split_date), linestyle="--")
plt.text(
        pd.to_datetime(calibration_split_date),
        monthly_counts["loan_count"].max() * 0.3,
        "Subtrain/Calibration Split",
        rotation=90,
        va="top",
    )

#makes it readable and not cut off when saved to file
plt.tight_layout()
plt.savefig(output_figure_path, dpi=150)
plt.close()

print("input:", input_path)
print("rows:", len(df))
print("months in plot:", len(monthly_counts))
print("wrote:")
print(" -", output_table_path)
print(" -", output_figure_path)