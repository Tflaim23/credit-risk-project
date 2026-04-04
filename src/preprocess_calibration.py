# Since the split was mostly the same for train/test as subtrain/calibration this likely won't be too different either
# Basically everything was the same and just copy and pasted with name changes 
# The only real difference is the original preprocess only fit on train and transform on test
# this one I needed to fit on subtrain then transform onto calibration and test
#thats basically the only reason I couldn't just create a new config for the original script

#I may have been able to just use a new config for the previous split step but I didn't realize how similar it was going to be

# Models can't deal with missing values and text categories so I need to make all the data numeric and fill in the missing values 
# I also want to add some flags for missing values in case they are informative
#use sklearn ColumnTransformer to apply different transformations to numeric and categorical features
# save the fitted preprocessor object to reuse it in the modeling stage
import argparse
from pathlib import Path
# joblib to save the fitted preprocessor object to reuse it 
import joblib
import pandas as pd
#deal with all the 0s with npz hopefully
import scipy.sparse as sp
import yaml

#all sklearmn imports for the preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


# Get feature col from the file that is just the saved feature set and save it as a list
def get_feature_list(feature_set_path: str) -> list[str]:
    feature_df = pd.read_csv(feature_set_path)
    return feature_df["feature"].tolist()


# Need to split features into quant and qual because they need different transformations
#feature hint just means it will return 2 lists 
def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    
    feature_df = df[features]
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in features if col not in numeric_cols]
    #alphabetical sort
    numeric_cols = sorted(numeric_cols)
    categorical_cols = sorted(categorical_cols)

    return numeric_cols, categorical_cols

def add_numeric_missing_flags(
    subtrain_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    subtrain_copy = subtrain_df.copy()
    calibration_copy = calibration_df.copy()
    test_copy = test_df.copy()

    flag_cols = []

    for col in numeric_cols:
        # Only checks numeric cols in training data just in case of leakage
        # if one is missing in the col it gets marked true
        if subtrain_copy[col].isna().any():
            flag_col = f"{col}_missing_flag"
            #0/1 col for any indicator that got marked true for missing values
            # where for each obs 1 means missing and 0 means not missing
            #This time for test and train since test no longer effects the flag creation
            subtrain_copy[flag_col] = subtrain_copy[col].isna().astype(int)
            calibration_copy[flag_col] = calibration_copy[col].isna().astype(int)
            test_copy[flag_col] = test_copy[col].isna().astype(int)

            flag_cols.append(flag_col)

    flag_cols = sorted(flag_cols)

    return subtrain_copy, calibration_copy, test_copy, flag_cols

#three list inputs and returns a ColumnTransformer from sklearn
#ColumnTransformer is something that enables functions to be applied to different cols simultaneously
def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    flag_cols: list[str],
) -> ColumnTransformer:
   
#pipelines are sequences of steps
#These two pipes fill in missing values for numeric and categorical cols respectively
#For numeric it is filled with median and for categorical it is filled with the most frequent value
#categorical also has onehot which gives it a 1 in its category and 0 in all others so model can deal with them

# (NEW) add a scaler to the pipeline because I think one of the problems with the model was that some features had way bigger ranges than others
#  I should have down this to begin with but it should help with that by standardizing the numeric features
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    #each transformer is name, transformer, and the cols it applies to
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
            ("flag", "passthrough", flag_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    subtrain_input_path = cfg["subtrain_input_path"]
    calibration_input_path = cfg["calibration_input_path"]
    test_input_path = cfg["test_input_path"]
    feature_set_path = cfg["feature_set_path"]

    target_col = cfg.get("target_col", "y")

    preprocessor_output_path = cfg["preprocessor_output_path"]
    x_subtrain_output_path = cfg["x_subtrain_output_path"]
    x_calibration_output_path = cfg["x_calibration_output_path"]
    x_test_output_path = cfg["x_test_output_path"]

    y_subtrain_output_path = cfg["y_subtrain_output_path"]
    y_calibration_output_path = cfg["y_calibration_output_path"]
    y_test_output_path = cfg["y_test_output_path"]

    feature_names_output_path = cfg["feature_names_output_path"]
    manifest_output_path = cfg["manifest_output_path"]

    subtrain_df = pd.read_parquet(subtrain_input_path)
    calibration_df = pd.read_parquet(calibration_input_path)
    test_df = pd.read_parquet(test_input_path)

    features = get_feature_list(feature_set_path)

    needed_cols = features + [target_col]

    #verify that all feature and target columns exist in all dataframes
    missing_subtrain = [col for col in needed_cols if col not in subtrain_df.columns]
    missing_calibration = [col for col in needed_cols if col not in calibration_df.columns]
    missing_test = [col for col in needed_cols if col not in test_df.columns]

    if missing_subtrain:
        raise ValueError(f"missing columns in subtrain file: {missing_subtrain}")
    if missing_calibration:
        raise ValueError(f"missing columns in calibration file: {missing_calibration}")
    if missing_test:
        raise ValueError(f"missing columns in test file: {missing_test}")

    numeric_cols, categorical_cols = split_feature_types(subtrain_df, features)

    subtrain_work = subtrain_df[needed_cols].copy()
    calibration_work = calibration_df[needed_cols].copy()
    test_work = test_df[needed_cols].copy()

    # Add missing flags before fitting the transformer.
    subtrain_work, calibration_work, test_work, flag_cols = add_numeric_missing_flags(
        subtrain_df=subtrain_work,
        calibration_df=calibration_work,
        test_df=test_work,
        numeric_cols=numeric_cols,
    )
    #calling the functiion to return the column transformer
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        flag_cols=flag_cols,
    )

    #Only fit to subtrain so it can save median sd mean and other values to avoid leakage and then just transform on calibration and test
    X_subtrain = preprocessor.fit_transform(subtrain_work)
    X_calibration = preprocessor.transform(calibration_work)
    X_test = preprocessor.transform(test_work)

    y_subtrain = subtrain_work[target_col].astype(int).copy()
    y_calibration = calibration_work[target_col].astype(int).copy()
    y_test = test_work[target_col].astype(int).copy()

    #mostly for onehot since it breaks up categorical cols into multiple cols
    feature_names = preprocessor.get_feature_names_out()
    feature_names_df = pd.DataFrame({"feature_name": feature_names})

    manifest = pd.DataFrame(
        [
            {
                "subtrain_rows": len(subtrain_work),
                "calibration_rows": len(calibration_work),
                "test_rows": len(test_work),
                "original_feature_count": len(features),
                "numeric_feature_count": len(numeric_cols),
                "categorical_feature_count": len(categorical_cols),
                "numeric_missing_flag_count": len(flag_cols),
                "transformed_feature_count": len(feature_names),
                "subtrain_bad_rate": round(float(y_subtrain.mean()), 2),
                "calibration_bad_rate": round(float(y_calibration.mean()), 2),
                "test_bad_rate": round(float(y_test.mean()), 2),
            }
        ]
    )

    ensure_parent_dir(preprocessor_output_path)
    ensure_parent_dir(x_subtrain_output_path)
    ensure_parent_dir(x_calibration_output_path)
    ensure_parent_dir(x_test_output_path)
    ensure_parent_dir(y_subtrain_output_path)
    ensure_parent_dir(y_calibration_output_path)
    ensure_parent_dir(y_test_output_path)
    ensure_parent_dir(feature_names_output_path)
    ensure_parent_dir(manifest_output_path)

    #save the fitted preprocessor object to reuse it in the modeling stage 
    joblib.dump(preprocessor, preprocessor_output_path)
    #save the fitted preprocessor object to reuse it in the modeling stage 
    sp.save_npz(x_subtrain_output_path, X_subtrain)
    sp.save_npz(x_calibration_output_path, X_calibration)
    sp.save_npz(x_test_output_path, X_test)

    y_subtrain.to_csv(y_subtrain_output_path, index=False)
    y_calibration.to_csv(y_calibration_output_path, index=False)
    y_test.to_csv(y_test_output_path, index=False)

    feature_names_df.to_csv(feature_names_output_path, index=False)
    manifest.to_csv(manifest_output_path, index=False)

    print("Subtrain input:", subtrain_input_path)
    print("Calibration input:", calibration_input_path)
    print("Test input:", test_input_path)
    print("Original feature count:", len(features))
    print("Numeric feature count:", len(numeric_cols))
    print("Categorical feature count:", len(categorical_cols))
    print("Numeric missing flag count:", len(flag_cols))
    print("Transformed subtrain shape:", X_subtrain.shape)
    print("Transformed calibration shape:", X_calibration.shape)
    print("Transformed test shape:", X_test.shape)
    print("Wrote:")
    print(" -", preprocessor_output_path)
    print(" -", x_subtrain_output_path)
    print(" -", x_calibration_output_path)
    print(" -", x_test_output_path)
    print(" -", y_subtrain_output_path)
    print(" -", y_calibration_output_path)
    print(" -", y_test_output_path)
    print(" -", feature_names_output_path)
    print(" -", manifest_output_path)


if __name__ == "__main__":
    main()