# I learned logistic regression is the best model to try first
# coefficients are easy to interpret 
# positive means that feature increases default risk, negative means it decreases it
#Lots of notes in this script (and likely those to come after) because a lot of it has to do with the sklearn package, even though I am overall new to python I am especially new to the package
import argparse
from pathlib import Path

#Saves py objects just like last script
import joblib
import pandas as pd
import scipy.sparse as sp
import yaml
from sklearn.linear_model import LogisticRegression
#These are used to determine performance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

#Returns the y values as a series
#iloc, : means gets all rows and and all y values are in col 0 the first one
def load_target_csv(path: str) -> pd.Series:

    df = pd.read_csv(path)
    return df.iloc[:, 0].astype(int)

# 3 inputs are: 
#actual is the true 0/1 bad outcome in test data (Same 1 means they defaulted, 0 means they didn't)
# predicted label is 0/1 prediction the model makes
# predicted probability the model assigns to each row (loan) for defalting
def compute_metrics(actual: pd.Series, predicted_label: pd.Series, predicted_prob: pd.Series) -> dict:
   
   #ROund everything to 5 decimals for accuracy
    metrics = {
        # Roc Auc is Receiver Operating Characteristic - Area Under Curve
        # This metric is to see how well the model ranks borrowers by risk (Putting the bad borrowers higher than the good ones when sorted by prob)
        # 1 means it perfectly ranks everyone, 0.5 means it is random
        #Roc auc score itself means if you take one random defaulter and one random non defaulter, what is the probability the model assigns a higher default probability to the defaulter
        "roc_auc": round(float(roc_auc_score(actual, predicted_prob)), 5),
        #Pr auc is Precision-Recall Area Under Curve
        # Since defaults are more rare this one is helpful to have as well
        # For this one it is the probability of when it flags a loan as bad is it actually bad
        # Since event rate (default rate) was 0.2008 that's the default instead of 0.5
        # Any better than the default rate beats random, 1 is perfect
        "pr_auc": round(float(average_precision_score(actual, predicted_prob)), 5),
        # More simple sklearn metric just what % of the time was the prediction right
        # I think it is probably not as helpful because if it just says 0% bad rate then it has 80% accuracy
        # I'd say a vacuum it's not good with imbalenced data but with the other metrics it could be good
        "accuracy": round(float(accuracy_score(actual, predicted_label)), 5),
        # f1 is only high when both are true:
        # High % of bad flagged loans are actually bad (precision)
        # High % of bad loans are flagged as bad (recall)
        "f1": round(float(f1_score(actual, predicted_label)), 5),
        # cross-entropy penalizes confident wrong predictons more than less confident predictions
        # 0 is perfect
        "log_loss": round(float(log_loss(actual, predicted_prob)), 5),
        # Mse of predictions and outcomes: Sum((actual - predicted_prob)^2) / n
        # Perfect would be 0 but since probability is continuous and actual is discrete 0/1 it can't be
        # Generally lower is better, models just have to be minimized against themselves rather than a strict threshhold I think
        "brier_score": round(float(brier_score_loss(actual, predicted_prob)), 5),
    }
    return metrics

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
    coefficients_output_path = cfg["coefficients_output_path"]

    #Random seed like with made up data from earlier, just chose my favorite number
    random_state = int(cfg.get("random_state", 23))
    # Lower C means coefficents are pushed to 0 more agressively, its like a lasso or ridge penalty term but inverse
    # Learned this in my SRM class
    c_value = float(cfg.get("c_value", 1.0))
    #How many tries the model gets to fit coefficents before it gives up, I set it high just to make sure it converges
    max_iter = int(cfg.get("max_iter", 1000))
    # above this it predicts bad, starting at 0.5 but I will probably move this (likely down)
    threshold = float(cfg.get("threshold", 0.50))

    X_train = sp.load_npz(x_train_path)
    X_test = sp.load_npz(x_test_path)

    y_train = load_target_csv(y_train_path)
    y_test = load_target_csv(y_test_path)

    feature_names = pd.read_csv(feature_names_path)["feature_name"].tolist()

    # if number of cols in training matrix isn't the number of feature names then that's no good
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature name count {len(feature_names)} does not match X_train columns {X_train.shape[1]}"
        )

    #setting up model settings for sklearn model
    model = LogisticRegression(
        #Ridge penalty
        penalty="l2",
        #set before in the script just another input for how aggresive the ridge is
        C=c_value,
        # Just one of the solvers sklearn uses it said saga is fast for large datasets but idk
        solver="saga",
        # Earlier in the script how many tried it gets to fit
        max_iter=max_iter,
        # random, 23 set seed
        random_state=random_state,
    )
    #actually fitting parameters to the datatsets
    model.fit(X_train, y_train)

    #returns two cols prob of 0 good and 1 bad, cols will sum to 1
    # [:,1] gets every row and gets just the prob of (bad) default in col index 1
    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    
    #true (predicting default) goes to 1
    train_label = (train_prob >= threshold).astype(int)
    test_label = (test_prob >= threshold).astype(int)

    #same explained function from earlier in script with all parts explained in detail
    train_metrics = compute_metrics(
        actual=y_train,
        predicted_label=train_label,
        predicted_prob=train_prob,
    )

    test_metrics = compute_metrics(
        actual=y_test,
        predicted_label=test_label,
        predicted_prob=test_prob,
    )

    #2 row df: one train one test 
    #** gets all the values out of the dictionary so they can be used individually
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

    #one row per test loan, shows actual outcome and then the model's prob and label
    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "predicted_prob_bad": test_prob,
            "predicted_label": test_label,
        }
    )

    #Saves the model's coeffecient for each feature
    #Ravel makes it 1D df
    coef_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": model.coef_.ravel(),
        }
    )

    #Want to see the most impactful regardless of neg/pos corr
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    ensure_parent_dir(model_output_path)
    ensure_parent_dir(metrics_output_path)
    ensure_parent_dir(predictions_output_path)
    ensure_parent_dir(coefficients_output_path)
    #save the whole model
    joblib.dump(model, model_output_path)

    metrics_df.to_csv(metrics_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)
    coef_df.to_csv(coefficients_output_path, index=False)

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
    print(" -", coefficients_output_path)


if __name__ == "__main__":
    main()