#(NEW) I ran this originally and got just the probability in the two output files
# Policy uses EL = PD * LGD * EAD where (expected loss) = (Probability of default)*(Loss given default)*(exposure at default)
#Loss given default would be a seperate metric within a company to detrimne what % of the loss they actually take on so I will just use a constant
#what I need to add is EAD which would just be the size of the loan or original loan amount


#Need a script that creates the exact policy input tables 
#for every loan in subtrain and test save the actual bad outcome and the boosted tree model’s predicted probability of default
#I am using subtrain because even though calibration did not work and was rejected, it would still introduce some level of bias and there is enough data in subtrain that it works fine
#one file for subtrain, one for test and then a manifest with all the info to keep it clean
# this is so the policy layer can choose acceptance thresholds on subtrain and then evaluate those same thresholds on untouched test

import argparse
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sp
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

#first col
def load_target_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[:, 0].astype(int)

def make_prediction_table(
    raw_df: pd.DataFrame,
    actual: pd.Series,
    predicted_prob,
    dataset_name: str,
    ead_column: str,
) -> pd.DataFrame:
    
    if ead_column not in raw_df.columns:
        raise ValueError(f"{ead_column}")

    #Basically only change is adding the EAD column which will always be loan amount
    out = pd.DataFrame(
        {
            "actual_bad": actual,
            "predicted_prob_bad": predicted_prob,
            "predicted_good_prob": 1.0 - predicted_prob,
            "ead": raw_df[ead_column].astype(float).values,
            "dataset": dataset_name,
        }
    )

    return out

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_path = cfg["model_path"]

    #to get EAD
    raw_subtrain_path = cfg["raw_subtrain_path"]
    raw_test_path = cfg["raw_test_path"]

    x_subtrain_path = cfg["x_subtrain_path"]
    y_subtrain_path = cfg["y_subtrain_path"]

    x_test_path = cfg["x_test_path"]
    y_test_path = cfg["y_test_path"]

    subtrain_output_path = cfg["subtrain_output_path"]
    test_output_path = cfg["test_output_path"]
    manifest_output_path = cfg["manifest_output_path"]

    model = joblib.load(model_path)

    X_subtrain = sp.load_npz(x_subtrain_path)
    X_test = sp.load_npz(x_test_path)

    y_subtrain = load_target_csv(y_subtrain_path)
    y_test = load_target_csv(y_test_path)

    ead_column = cfg.get("ead_column", "loan_amnt")
    
    raw_subtrain = pd.read_parquet(raw_subtrain_path)
    raw_test = pd.read_parquet(raw_test_path)

    subtrain_prob = model.predict_proba(X_subtrain)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    subtrain_df = make_prediction_table(
        raw_df=raw_subtrain,
        actual=y_subtrain,
        predicted_prob=subtrain_prob,
        dataset_name="subtrain",
        ead_column=ead_column,
    )

    test_df = make_prediction_table(
        raw_df=raw_test,
        actual=y_test,
        predicted_prob=test_prob,
        dataset_name="test",
        ead_column=ead_column,
    )

    manifest = pd.DataFrame(
        [
            {
                "dataset": "subtrain",
                "rows": len(subtrain_df),
                "actual_bad_rate": round(float(subtrain_df["actual_bad"].mean()), 2),
                "avg_predicted_prob_bad": round(float(subtrain_df["predicted_prob_bad"].mean()), 2),
                "avg_ead": round(float(subtrain_df["ead"].mean()), 2),
            },
            {
                "dataset": "test",
                "rows": len(test_df),
                "actual_bad_rate": round(float(test_df["actual_bad"].mean()), 2),
                "avg_predicted_prob_bad": round(float(test_df["predicted_prob_bad"].mean()), 2),
                "avg_ead": round(float(test_df["ead"].mean()), 2),
            },
        ]
    )

    ensure_parent_dir(subtrain_output_path)
    ensure_parent_dir(test_output_path)
    ensure_parent_dir(manifest_output_path)

    subtrain_df.to_csv(subtrain_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    manifest.to_csv(manifest_output_path, index=False)

    print("model:", model_path)
    print("Subtrain rows:", len(subtrain_df))
    print("Test rows:", len(test_df))
    print()
    print(manifest.to_string(index=False))
    print()
    print("wrote:")
    print(" -", subtrain_output_path)
    print(" -", test_output_path)
    print(" -", manifest_output_path)


if __name__ == "__main__":
    main()