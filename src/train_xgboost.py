#Overall this should be similar to the logreg script just now using xgboost to boost trees
import argparse
from pathlib import Path

#Same to save model
import joblib
import pandas as pd
#same to deal with 0's in matrix cols
import scipy.sparse as sp
import yaml
#all the same metric imports for comparison
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
#Boosted trees, this is in my modeling requirments and installed with pip install -r requirements/modeling.txt
from xgboost import XGBClassifier

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


#Same read csv and get all rows of the first col
def load_target_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[:, 0].astype(int)


#Same exact thing as in logreg check that script for notes on each metric
def compute_metrics(actual: pd.Series, predicted_label, predicted_prob) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(actual, predicted_prob)), 5),
        "pr_auc": round(float(average_precision_score(actual, predicted_prob)), 5),
        "accuracy": round(float(accuracy_score(actual, predicted_label)), 5),
        "f1": round(float(f1_score(actual, predicted_label)), 5),
        "log_loss": round(float(log_loss(actual, predicted_prob)), 5),
        "brier_score": round(float(brier_score_loss(actual, predicted_prob)), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    x_train_path = cfg["x_train_path"]
    x_test_path = cfg["x_test_path"]
    y_train_path = cfg["y_train_path"]
    y_test_path = cfg["y_test_path"]
    feature_names_path = cfg["feature_names_path"]

    model_output_path = cfg["model_output_path"]
    metrics_output_path = cfg["metrics_output_path"]
    predictions_output_path = cfg["predictions_output_path"]
    importances_output_path = cfg["importances_output_path"]

    #all explained in yaml
    random_state = int(cfg.get("random_state", 23))
    n_estimators = int(cfg.get("n_estimators", 300))
    learning_rate = float(cfg.get("learning_rate", 0.05))
    max_depth = int(cfg.get("max_depth", 6))
    min_child_weight = float(cfg.get("min_child_weight", 50))
    subsample = float(cfg.get("subsample", 0.8))
    colsample_bytree = float(cfg.get("colsample_bytree", 0.8))
    reg_lambda = float(cfg.get("reg_lambda", 1.0))
    threshold = float(cfg.get("threshold", 0.50))
    n_jobs = int(cfg.get("n_jobs", -1))

    X_train = sp.load_npz(x_train_path)
    X_test = sp.load_npz(x_test_path)

    y_train = load_target_csv(y_train_path)
    y_test = load_target_csv(y_test_path)
    #py list of all features
    feature_names = pd.read_csv(feature_names_path)["feature_name"].tolist()

    #Same thing as in logreg making sure # of cols in training matrix = # of feature names
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name count {len(feature_names)} does not match X_train columns {X_train.shape[1]}"
        )

    model = XGBClassifier(
        # logistic regression for binary classification, output probability
        objective="binary:logistic",
        # negative log-likelihood,  penalizes confident wrong predictions 
        eval_metric="logloss",
        #Fast histogram based tree building algo
        tree_method="hist",
        #all below explained in yaml
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    #xgboost fit function on model to build the trees on the training data
    model.fit(X_train, y_train)

    #returns two cols prob of 0 good and 1 bad, cols will sum to 1
    # [:,1] gets every row and gets just the prob of (bad) default in col index 1
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    #loans over threshold get marked default (1)
    train_label = (train_prob >= threshold).astype(int)
    test_label = (test_prob >= threshold).astype(int)

    #Calls function to get all my metrics
    train_metrics = compute_metrics(y_train, train_label, train_prob)
    test_metrics = compute_metrics(y_test, test_label, test_prob)

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "train",
                "rows": int(X_train.shape[0]),
                "threshold": threshold,
                **train_metrics,
            },
            {
                "dataset": "test",
                "rows": int(X_test.shape[0]),
                "threshold": threshold,
                **test_metrics,
            },
        ]
    )

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted_prob_bad": test_prob,
            "predicted_label": test_label,
        }
    )
    # Only new thing 
    # Boosted trees use importance of features rather than coef
    # Coefficients like in logreg tell the direction and magnitude of each feature's effect, (positive = more risk, negative = less risk). Boosted trees use importance
    # Importance only tells you which were used the most for splits, not whether they had a positive or negative effect
    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    ensure_parent_dir(model_output_path)
    ensure_parent_dir(metrics_output_path)
    ensure_parent_dir(predictions_output_path)
    ensure_parent_dir(importances_output_path)

    # All 300 trees with all their splits, leaf values, and learned parameters get frozen
    joblib.dump(model, model_output_path)

    #same copy and pasted from logreg just swapping coef for importance
    #index just means it doesnt number the rows as its own col I forget to put that
    metrics_df.to_csv(metrics_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)
    importance_df.to_csv(importances_output_path, index=False)

    print("Train matrix shape:", X_train.shape)
    print("Test matrix shape:", X_test.shape)
    print("Threshold:", threshold)
    print()
    print("Train metrics")
    print(metrics_df.loc[metrics_df["dataset"] == "train"].to_string(index=False))
    print()
    print("Test metrics")
    print(metrics_df.loc[metrics_df["dataset"] == "test"].to_string(index=False))
    print()
    print("Wrote:")
    print(" -", model_output_path)
    print(" -", metrics_output_path)
    print(" -", predictions_output_path)
    print(" -", importances_output_path)


if __name__ == "__main__":
    main()