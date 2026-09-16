import pandas as pd
import os


# ==============================
# PATHS
# ==============================

INPUT_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/processed/churn_processed.csv"


# ==============================
# LOAD DATA
# ==============================

def load_data():
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset loaded successfully: {df.shape}")
    return df


# ==============================
# CLEAN DATA
# ==============================

def clean_data(df):

    # Remove unnecessary customer ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Handle missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Convert target variable
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove any remaining duplicate rows
    df = df.drop_duplicates()

    return df


# ==============================
# SAVE PROCESSED DATA
# ==============================

def save_data(df):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Processed dataset saved to: {OUTPUT_PATH}")
    print(f"Processed dataset shape: {df.shape}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("INTELLIPREDICT DATA PREPROCESSING")
    print("=" * 50)

    data = load_data()

    print("\nOriginal columns:")
    print(data.columns.tolist())

    data = clean_data(data)

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nTarget distribution:")
    print(data["Churn"].value_counts())

    save_data(data)

    print("\nPreprocessing completed successfully!")