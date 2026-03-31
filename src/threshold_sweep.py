#I want to make a script that will test thresholds for both models that I can just swap with yaml config
# This will test all threshokld values from start to stop with step all in yaml
# make decisions using train data only and apply them to test to avoid leakage
import argparse
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sp
import yaml
# Getting the metrics that are determined by the threshold 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

#Same block that reads y_train or y_test and returns the first column (true outcome) as 0s and 1s
def load_target_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[:, 0].astype(int)

#Build thresholds to test
def make_threshold_list(start: float, stop: float, step: float) -> list[float]:
    
    thresholds = []
    current = start
    #while the current threshold is less than the stop, add it to the list then take a step and check again
    while current <= stop + 0.01:
        thresholds.append(round(current, 2))
        current += step

    return thresholds

def compute_threshold_metrics(actual: pd.Series, predicted_prob, threshold: float) -> dict:
    # At a threshold turns predicted defualts to 1s and non-defaults to 0s
    predicted_label = (predicted_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        #proportion of accurate predictions out of all predictions
        "accuracy": round(float(accuracy_score(actual, predicted_label)), 6),
        # Of all preicted default, how many were actually default
        "precision": round(float(precision_score(actual, predicted_label, zero_division=0)), 6),
        # Off all defaults, how many were predicted as default
        "recall": round(float(recall_score(actual, predicted_label, zero_division=0)), 6),
        # Gives a score that balances the last two, 1 is perfect
        "f1": round(float(f1_score(actual, predicted_label, zero_division=0)), 6),
        # What % of all loans does it predict default
        "predicted_bad_rate": round(float(predicted_label.mean()), 6),
    }


def build_threshold_table(actual: pd.Series, predicted_prob, thresholds: list[float]) -> pd.DataFrame:

    rows = []
    #loops through every threshold and gets the metrics and adds it to the rows list
    for threshold in thresholds:
        row = compute_threshold_metrics(actual, predicted_prob, threshold)
        rows.append(row)

    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_path = cfg["model_path"]
    x_train_path = cfg["x_train_path"]
    x_test_path = cfg["x_test_path"]
    y_train_path = cfg["y_train_path"]
    y_test_path = cfg["y_test_path"]

    train_output_path = cfg["train_output_path"]
    test_output_path = cfg["test_output_path"]
    summary_output_path = cfg["summary_output_path"]

    model_name = cfg.get("model_name", "model")
    #whatever is in yaml to max out
    optimize_metric = cfg.get("optimize_metric", "f1")

    threshold_start = float(cfg.get("threshold_start", 0.05))
    threshold_stop = float(cfg.get("threshold_stop", 0.70))
    threshold_step = float(cfg.get("threshold_step", 0.01))

    # Load the model in yaml (either logreg or xgb)
    model = joblib.load(model_path)

    X_train = sp.load_npz(x_train_path)
    X_test = sp.load_npz(x_test_path)

    y_train = load_target_csv(y_train_path)
    y_test = load_target_csv(y_test_path)

    #Gets prob of default for each loan 
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    thresholds = make_threshold_list(threshold_start, threshold_stop, threshold_step)

    train_table = build_threshold_table(y_train, train_prob, thresholds)
    test_table = build_threshold_table(y_test, test_prob, thresholds)

    train_table["dataset"] = "train"
    test_table["dataset"] = "test"
    train_table["model_name"] = model_name
    test_table["model_name"] = model_name

    # Choose the best threshold using TRAIN only based on the optimize metric (f1 or whatever is in yaml) and then find the corresponding metrics for that threshold in test
    best_train_row = train_table.sort_values(
        by=[optimize_metric],
        ascending=[False],
    ).iloc[0]

    chosen_threshold = float(best_train_row["threshold"])
    #Gets the row in the test table that corresponds to the chosen threshold from train and gets the metrics for that threshold in test
    chosen_test_row = test_table.loc[test_table["threshold"] == chosen_threshold].iloc[0]

    #Summary gets me everything I need like which model and which metric was optimized
    #what threshold was chosen from train
    #all five metrics on train at that threshold
    #all five metrics on test at that same threshold
    summary_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "optimize_metric": optimize_metric,
                "chosen_threshold_from_train": chosen_threshold,
                "train_accuracy": best_train_row["accuracy"],
                "train_precision": best_train_row["precision"],
                "train_recall": best_train_row["recall"],
                "train_f1": best_train_row["f1"],
                "train_predicted_bad_rate": best_train_row["predicted_bad_rate"],
                "test_accuracy_at_chosen_threshold": chosen_test_row["accuracy"],
                "test_precision_at_chosen_threshold": chosen_test_row["precision"],
                "test_recall_at_chosen_threshold": chosen_test_row["recall"],
                "test_f1_at_chosen_threshold": chosen_test_row["f1"],
                "test_predicted_bad_rate_at_chosen_threshold": chosen_test_row["predicted_bad_rate"],
            }
        ]
    )

    ensure_parent_dir(train_output_path)
    ensure_parent_dir(test_output_path)
    ensure_parent_dir(summary_output_path)

    train_table.to_csv(train_output_path, index=False)
    test_table.to_csv(test_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("Model:", model_name)
    print("Model path:", model_path)
    print("Threshold count:", len(thresholds))
    print("Optimize metric:", optimize_metric)
    print("Chosen threshold from train:", chosen_threshold)
    print()
    print("Summary")
    print(summary_df.to_string(index=False))
    print()
    print("Wrote:")
    print(" -", train_output_path)
    print(" -", test_output_path)
    print(" -", summary_output_path)


if __name__ == "__main__":
    main()