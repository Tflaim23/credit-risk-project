#Visually I can just tell that the first run on boosted trees beats logreg 
# I figured it would be worthwhile to combined the metrics and sort it by best model to document my choice
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

#Copys both metric dfs for boosted and logreg, adds a col called "model_name" and fills it with the name for test and train
#model names in yaml and this gets called for both models later 
def add_model_name(metrics_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = metrics_df.copy()
    out["model_name"] = model_name
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    logreg_metrics_path = cfg["logreg_metrics_path"]
    xgboost_metrics_path = cfg["xgboost_metrics_path"]
    output_path = cfg["output_path"]

    logreg_name = cfg.get("logreg_name", "logistic_regression_v2")
    xgboost_name = cfg.get("xgboost_name", "xgboost_v1")

    logreg_df = pd.read_csv(logreg_metrics_path)
    xgboost_df = pd.read_csv(xgboost_metrics_path)

    logreg_df = add_model_name(logreg_df, logreg_name)
    xgboost_df = add_model_name(xgboost_df, xgboost_name)

    #pdoncat stacks dfs, stacking the two metric dfs with the extra col of their name
    #ignore_index makes it so it reindexes after the stack
    comparison_df = pd.concat([logreg_df, xgboost_df], ignore_index=True)

    #put modelname first so it is easier to read
    ordered_cols = [
        "model_name",
        "dataset",
        "rows",
        "threshold",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
        "log_loss",
        "brier_score",
    ]
    comparison_df = comparison_df[ordered_cols]

    # Sort so that test rows appear first and strongest models come first
    # Every test becomes 0 and every train becomes 1 so test will come first
    comparison_df["dataset_sort"] = comparison_df["dataset"].map({"test": 0, "train": 1})
    comparison_df = comparison_df.sort_values(
        # sort by these three columns in this order of priority
        by=["dataset_sort", "roc_auc", "pr_auc"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    #drop the 0s and 1s used to put test first 
    comparison_df = comparison_df.drop(columns=["dataset_sort"])

    ensure_parent_dir(output_path)
    comparison_df.to_csv(output_path, index=False)

    print("Logistic metrics input:", logreg_metrics_path)
    print("XGBoost metrics input:", xgboost_metrics_path)
    print("Output:", output_path)
    print()
    print("Model comparison table")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()