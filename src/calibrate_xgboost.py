#Script that actually calibrates the model
#Training the original boosted trees model is to be good at ranking the loans risk of default most to least risky 
#calibration is needed to make them line up with reality so the probabilty spits out for each loan actually reflect the odds of defaulting
#acomplish this by taking taking the model’s original probability predictions and adjusting them using real outcomes so the numbers it predicts match what actually happens\

#basically the bins let me compare predicted probabilities to actual outcomes in groups, and sigmoid calibration adjusts those probabilities so they line up better
#needed to be on a seperate slice of data than the one used to train the original model to so I can learn the correction without biasing it and still keep test untouched for a final honest check (also avoid data leakage)
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import yaml
#sk learn calibration tools
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
#Used to freeze original model and not refit it during calibration
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    
#Read the saved target CSV and return the first column
def load_target_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[:, 0].astype(int)

#similar to modeling scripts but swap f1 and accuracy for predicted prob and actual bad rate 
# thats bc calibration is more about if the predictions line up with reality while threshold was more about yes/no
def compute_probability_metrics(actual: pd.Series, predicted_prob) -> dict:

    return {
        "roc_auc": round(float(roc_auc_score(actual, predicted_prob)), 5),
        "pr_auc": round(float(average_precision_score(actual, predicted_prob)), 5),
        "log_loss": round(float(log_loss(actual, predicted_prob)), 5),
        "brier_score": round(float(brier_score_loss(actual, predicted_prob)), 5),
        "avg_predicted_prob": round(float(pd.Series(predicted_prob).mean()), 5),
        "actual_bad_rate": round(float(actual.mean()), 5),
    }

#builds the points used for the calibration curve
def build_calibration_curve_table(
    y_true: pd.Series,
    y_prob,
    model_name: str,
    dataset_name: str,
    n_bins: int,
    strategy: str,
) -> pd.DataFrame:
    #sklearn function that puts the probability into two bins
    #returns fraction of positive in each bin and the mean predicted prob in each bin
    prob_true, prob_pred = calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        n_bins=n_bins,
        strategy=strategy,
    )
    #returns a table with the two names and the two arrays from last function 
    return pd.DataFrame(
        {
            "model_name": model_name,
            "dataset": dataset_name,
            "bin_mean_predicted_prob": prob_pred,
            "bin_fraction_positive": prob_true,
        }
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    base_model_path = cfg["base_model_path"]

    x_calibration_path = cfg["x_calibration_path"]
    y_calibration_path = cfg["y_calibration_path"]

    x_test_path = cfg["x_test_path"]
    y_test_path = cfg["y_test_path"]

    calibrated_model_output_path = cfg["calibrated_model_output_path"]
    metrics_output_path = cfg["metrics_output_path"]
    predictions_output_path = cfg["predictions_output_path"]
    curve_table_output_path = cfg["curve_table_output_path"]
    curve_figure_output_path = cfg["curve_figure_output_path"]

    method = cfg.get("method", "sigmoid")
    n_bins = int(cfg.get("n_bins", 10))
    curve_strategy = cfg.get("curve_strategy", "quantile")

    base_model = joblib.load(base_model_path)
    #x features is in npz because of all the 0s and y is still just actual outcomes, everything same as usual just loading in everything from the congig dictionary 
    X_cal = sp.load_npz(x_calibration_path)
    y_cal = load_target_csv(y_calibration_path)

    X_test = sp.load_npz(x_test_path)
    y_test = load_target_csv(y_test_path)

    # have the model get uncalibrated probabilities of default for every row in the dfs
    cal_prob_uncal = base_model.predict_proba(X_cal)[:, 1]
    test_prob_uncal = base_model.predict_proba(X_test)[:, 1]

    # Calibrate the already fitted model (using frozen estimator) on the calibration slice only
    # use the sigmoid method or whatever is in config
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method=method,
    )
    #raw to calibrated probs being learned
    calibrated_model.fit(X_cal, y_cal)

    #Same as a couple lines up but now just using the calibrated model not the original
    cal_prob_calibrated = calibrated_model.predict_proba(X_cal)[:, 1]
    test_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    metrics_rows = []
    
    #dictionaries with model version and the dataset then inject them with the computed metrics from the earlier function with ** which unpacks it
    metrics_rows.append(
        {
            "model_version": "xgboost_subtrain_uncalibrated",
            "dataset": "calibration",
            **compute_probability_metrics(y_cal, cal_prob_uncal),
        }
    )

    metrics_rows.append(
        {
            "model_version": "xgboost_subtrain_calibrated",
            "dataset": "calibration",
            **compute_probability_metrics(y_cal, cal_prob_calibrated),
        }
    )

    metrics_rows.append(
        {
            "model_version": "xgboost_subtrain_uncalibrated",
            "dataset": "test",
            **compute_probability_metrics(y_test, test_prob_uncal),
        }
    )

    metrics_rows.append(
        {
            "model_version": "xgboost_subtrain_calibrated",
            "dataset": "test",
            **compute_probability_metrics(y_test, test_prob_calibrated),
        }
    )

    metrics_df = pd.DataFrame(metrics_rows)

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "uncalibrated_prob_bad": test_prob_uncal,
            "calibrated_prob_bad": test_prob_calibrated,
        }
    )

    curve_tables = []

    #calls function to build the df for both calibrated and uncalibrated
    curve_tables.append(
        build_calibration_curve_table(
            y_true=y_test,
            y_prob=test_prob_uncal,
            model_name="xgboost_subtrain_uncalibrated",
            dataset_name="test",
            n_bins=n_bins,
            strategy=curve_strategy,
        )
    )

    curve_tables.append(
        build_calibration_curve_table(
            y_true=y_test,
            y_prob=test_prob_calibrated,
            model_name="xgboost_subtrain_calibrated",
            dataset_name="test",
            n_bins=n_bins,
            strategy=curve_strategy,
        )
    )
    #combine
    curve_df = pd.concat(curve_tables, ignore_index=True)

    ensure_parent_dir(calibrated_model_output_path)
    ensure_parent_dir(metrics_output_path)
    ensure_parent_dir(predictions_output_path)
    ensure_parent_dir(curve_table_output_path)
    ensure_parent_dir(curve_figure_output_path)

    #save calibrated model
    joblib.dump(calibrated_model, calibrated_model_output_path)

    metrics_df.to_csv(metrics_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)
    curve_df.to_csv(curve_table_output_path, index=False)

    # reliability plot
    plt.figure(figsize=(8, 6))

    #loops for calibrated and uncalibrated
    for model_name in curve_df["model_name"].unique():
        temp = curve_df.loc[curve_df["model_name"] == model_name]
        #x is mean predicted prob and y is actual bad rate in each bin
        plt.plot(
            temp["bin_mean_predicted_prob"],
            temp["bin_fraction_positive"],
            marker="o",
            label=model_name,
        )

#target line for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect_calibration")
    plt.xlabel("predicted probability")
    plt.ylabel("bad rate")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_figure_output_path, dpi=150)
    plt.close()

    print("base model:", base_model_path)
    print("Calibration method:", method)
    print()
    print("Calibration/test metric comparison")
    print(metrics_df.to_string(index=False))
    print()
    print("Wrote:")
    print(" -", calibrated_model_output_path)
    print(" -", metrics_output_path)
    print(" -", predictions_output_path)
    print(" -", curve_table_output_path)
    print(" -", curve_figure_output_path)


if __name__ == "__main__":
    main()