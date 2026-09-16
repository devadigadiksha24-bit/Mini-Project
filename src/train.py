import os
import json
import time
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# INTELLIPREDICT - MODEL TRAINING PIPELINE
# ============================================================

print("=" * 70)
print("INTELLIPREDICT MLOPS - MODEL TRAINING")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/churn_processed.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

METRICS_PATH = "metrics.json"

COMPARISON_PATH = "model_comparison.csv"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MLflow CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "IntelliPredict-Churn"

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run data_preprocessing.py first."
    )


df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded successfully: "
    f"{df.shape[0]} rows, {df.shape[1]} columns"
)


# ============================================================
# VALIDATE TARGET
# ============================================================

if "Churn" not in df.columns:

    raise ValueError(
        "Target column 'Churn' was not found in the dataset."
    )


# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

if "customerID" in df.columns:

    df = df.drop(
        columns=["customerID"]
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


numerical_columns = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ============================================================
# PREPROCESSOR FUNCTION
# ============================================================

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                numeric_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )


    return preprocessor


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),


    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),


    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),


    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
}


# ============================================================
# VARIABLES FOR BEST MODEL
# ============================================================

best_model = None

best_model_name = None

best_f1 = -1

best_roc_auc = -1

all_results = []


# ============================================================
# TRAIN EACH MODEL
# ============================================================

for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)


    # --------------------------------------------------------
    # Create a NEW preprocessor for every model
    # --------------------------------------------------------

    preprocessor = create_preprocessor()


    # --------------------------------------------------------
    # Create ML pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    # --------------------------------------------------------
    # Start timer
    # --------------------------------------------------------

    start_time = time.time()


    # --------------------------------------------------------
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ====================================================
        # TRAIN MODEL
        # ====================================================

        pipeline.fit(
            X_train,
            y_train
        )


        # ====================================================
        # PREDICTIONS
        # ====================================================

        predictions = pipeline.predict(
            X_test
        )


        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]


        # ====================================================
        # CALCULATE METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )


        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )


        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )


        training_time = (
            time.time() - start_time
        )


        # ====================================================
        # LOG PARAMETERS
        # ====================================================

        mlflow.log_param(
            "model_name",
            model_name
        )


        mlflow.log_param(
            "test_size",
            0.20
        )


        mlflow.log_param(
            "random_state",
            42
        )


        # ====================================================
        # LOG METRICS
        # ====================================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )


        mlflow.log_metric(
            "precision",
            precision
        )


        mlflow.log_metric(
            "recall",
            recall
        )


        mlflow.log_metric(
            "f1_score",
            f1
        )


        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        mlflow.log_metric(
            "training_time",
            training_time
        )


        # ====================================================
        # LOG MODEL
        #
        # cloudpickle avoids the skops trust issue
        # encountered with numpy.dtype
        # ====================================================

        mlflow.sklearn.log_model(

            pipeline,

            name="model",

            serialization_format="cloudpickle"
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"Accuracy     : {accuracy:.4f}"
        )

        print(
            f"Precision    : {precision:.4f}"
        )

        print(
            f"Recall       : {recall:.4f}"
        )

        print(
            f"F1 Score     : {f1:.4f}"
        )

        print(
            f"ROC-AUC      : {roc_auc:.4f}"
        )

        print(
            f"Training Time: {training_time:.2f} seconds"
        )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    result = {

        "model": model_name,

        "accuracy": round(
            accuracy,
            4
        ),

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1_score": round(
            f1,
            4
        ),

        "roc_auc": round(
            roc_auc,
            4
        ),

        "training_time": round(
            training_time,
            4
        )
    }


    all_results.append(
        result
    )


    # ========================================================
    # BEST MODEL SELECTION
    #
    # Primary metric  : F1 Score
    # Secondary metric: ROC-AUC
    # ========================================================

    if (

        f1 > best_f1

        or

        (
            f1 == best_f1
            and roc_auc > best_roc_auc
        )

    ):

        best_f1 = f1

        best_roc_auc = roc_auc

        best_model = pipeline

        best_model_name = model_name


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print(
    f"\nModel comparison saved to: "
    f"{COMPARISON_PATH}"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

if best_model is None:

    raise RuntimeError(
        "No model was successfully trained."
    )


joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics = {

    "project": "IntelliPredict MLOps",

    "experiment": EXPERIMENT_NAME,

    "best_model": best_model_name,

    "best_f1_score": round(
        best_f1,
        4
    ),

    "best_roc_auc": round(
        best_roc_auc,
        4
    ),

    "models": all_results
}


with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        final_metrics,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL SELECTION")
print("=" * 70)

print(
    f"Best Model : {best_model_name}"
)

print(
    f"Best F1    : {best_f1:.4f}"
)

print(
    f"Best ROC-AUC: {best_roc_auc:.4f}"
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

print(
    f"Metrics saved to: {METRICS_PATH}"
)

print(
    f"Comparison saved to: {COMPARISON_PATH}"
)


print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)