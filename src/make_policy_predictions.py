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

# build one clean table with actual outcomes and predicted bad probabilities
# Need this for moving on to policy evalutation
def make_prediction_table(actual: pd.Series, predicted_prob, dataset_name: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "actual_bad": actual,
            "predicted_prob_bad": predicted_prob,
        }
    )
    #use dataset_name so I can do iut for subtrain and test
    out["dataset"] = dataset_name
    out["predicted_good_prob"] = 1 - out["predicted_prob_bad"]

    return out

def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_path = cfg["model_path"]

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

    subtrain_prob = model.predict_proba(X_subtrain)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    subtrain_df = make_prediction_table(
        actual=y_subtrain,
        predicted_prob=subtrain_prob,
        dataset_name="subtrain",
    )

    test_df = make_prediction_table(
        actual=y_test,
        predicted_prob=test_prob,
        dataset_name="test",
    )

    manifest = pd.DataFrame(
        [
            {
                "dataset": "subtrain",
                "rows": len(subtrain_df),
                "actual_bad_rate": round(float(subtrain_df["actual_bad"].mean()), 2),
                "avg_predicted_prob_bad": round(float(subtrain_df["predicted_prob_bad"].mean()), 2),
            },
            {
                "dataset": "test",
                "rows": len(test_df),
                "actual_bad_rate": round(float(test_df["actual_bad"].mean()), 2),
                "avg_predicted_prob_bad": round(float(test_df["predicted_prob_bad"].mean()), 2),
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