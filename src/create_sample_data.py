import os
from pathlib import Path

import numpy as np
import pandas as pd

# making test data to try before using real loan data
def main() -> None:
    # creating the same random sample, 23 is just a random seed its my favorite number
    rng = np.random.default_rng(seed=23)

    # number of fake loan records
    n = 2000

    # fake origination date column
    issue_dates = pd.date_range("2016-01-01", "2019-12-01", freq="MS")
    issue_d = rng.choice(issue_dates, size=n, replace=True)

    # fake random loan data
    loan_amnt = rng.integers(1000, 35000, size=n)
    annual_inc = rng.normal(loc=75000, scale=25000, size=n).clip(15000, 250000).round(0)
    dti = rng.normal(loc=15, scale=8, size=n).clip(0, 45).round(2)
    fico = rng.integers(600, 850, size=n)

    term = rng.choice([" 36 months", " 60 months"], size=n, p=[0.75, 0.25])
    grade = rng.choice(list("ABCDEFG"), size=n, p=[0.25, 0.20, 0.20, 0.15, 0.1, 0.06, 0.04])
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.40, 0.40, 0.20])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"],
        size=n,
        p=[0.50, 0.20, 0.10, 0.10, 0.10],
    )

    # creeate interest te based on grade and fico and some noise
    grade_to_base = {g: b for g, b in zip(list("ABCDEFG"), [2.5, 7.5, 12.5, 15.0, 17.5, 20.0, 25.0])}
    base_rate = np.array([grade_to_base[g] for g in grade])
    int_rate = (base_rate + (700 - fico) * 0.015 + rng.normal(0, 0.75, size=n)).clip(5, 35).round(2)

    # made up default probability
    pd_logit = (
        -3.0
        + (np.array([ord(g) - ord("A") for g in grade]) * 0.35)
        + (dti * 0.03)
        + ((700 - fico) * 0.01)
        + (term == " 60 months") * 0.35
    )
    pd_true = 1 / (1 + np.exp(-pd_logit))

    # find outcomes based on default prob
    default_flag = rng.binomial(n=1, p=pd_true, size=n)

    # use labels like lendingclub data does
    loan_status = np.where(default_flag == 1, "Charged Off", "Fully Paid")

    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(issue_d),
            "loan_amnt": loan_amnt,
            "int_rate": int_rate,
            "term": term,
            "grade": grade,
            "home_ownership": home_ownership,
            "purpose": purpose,
            "annual_inc": annual_inc,
            "dti": dti,
            "fico": fico,
            "loan_status": loan_status,
        }
    ).sort_values("issue_d")

    #add random missing values to 4 categories to make sure missingness works in the profile report
    df.loc[rng.random(n) < 0.05, "annual_inc"] = np.nan
    df.loc[rng.random(n) < 0.03, "dti"] = np.nan
    df.loc[rng.random(n) < 0.02, "fico"] = np.nan
    df.loc[rng.random(n) < 0.02, "home_ownership"] = pd.NA
    
    # write to csv
    out_dir = Path("data/sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "loans_sample.csv"
    df.to_csv(out_path, index=False)

    print(f"created: {out_path} (rows={len(df):,}, cols={df.shape[1]})")


if __name__ == "__main__":
    main()