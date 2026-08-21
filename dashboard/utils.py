import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_fraud_dataset.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "fraud_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the financial fraud detection dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained fraud detection model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at:\n{MODEL_PATH}\n\n"
            "Please train the fraud detection model first "
            "and save it as fraud_model.pkl inside the "
            "models folder."
        )

    model = joblib.load(MODEL_PATH)

    return model

# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor():
    """Load the preprocessing/scaler object used during training."""

    preprocessor_paths = [
        BASE_DIR / "models" / "preprocessor.pkl",
        BASE_DIR / "models" / "fraud_preprocessor.pkl",
        BASE_DIR / "models" / "scaler.pkl"
    ]

    for path in preprocessor_paths:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError(
        "Preprocessor not found.\n\n"
        "Expected one of these files inside the models folder:\n"
        "  - preprocessor.pkl\n"
        "  - fraud_preprocessor.pkl\n"
        "  - scaler.pkl\n"
    )
# ============================================================
# GET FRAUD COLUMN
# ============================================================

def get_fraud_column(df):
    """
    Find the fraud label column in the dataset.
    """

    possible_columns = [
        "Fraud_Label",
        "is_fraud",
        "Fraud",
        "fraud"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise KeyError(
        "Fraud column not found.\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# GET AMOUNT COLUMN
# ============================================================

def get_amount_column(df):
    """
    Find the transaction amount column.
    """

    possible_columns = [
        "Transaction_Amount",
        "amount",
        "Amount",
        "transaction_amount"
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    raise KeyError(
        "Transaction amount column not found.\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ============================================================
# FRAUD PERCENTAGE
# ============================================================

def fraud_percentage(df):
    """
    Calculate percentage of fraudulent transactions.
    """

    if df.empty:
        return 0.0

    fraud_column = get_fraud_column(df)

    fraud_count = (
        pd.to_numeric(
            df[fraud_column],
            errors="coerce"
        )
        .fillna(0)
        .eq(1)
        .sum()
    )

    return (
        fraud_count / len(df)
    ) * 100


# ============================================================
# TOTAL FRAUD AMOUNT
# ============================================================

def total_fraud_amount(df):
    """
    Calculate total amount involved in fraudulent transactions.
    """

    fraud_column = get_fraud_column(df)
    amount_column = get_amount_column(df)

    fraud_labels = pd.to_numeric(
        df[fraud_column],
        errors="coerce"
    ).fillna(0)

    amounts = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    ).fillna(0)

    return amounts.loc[
        fraud_labels == 1
    ].sum()


# ============================================================
# TOTAL LEGITIMATE AMOUNT
# ============================================================

def total_legitimate_amount(df):
    """
    Calculate total amount involved in legitimate transactions.
    """

    fraud_column = get_fraud_column(df)
    amount_column = get_amount_column(df)

    fraud_labels = pd.to_numeric(
        df[fraud_column],
        errors="coerce"
    ).fillna(0)

    amounts = pd.to_numeric(
        df[amount_column],
        errors="coerce"
    ).fillna(0)

    return amounts.loc[
        fraud_labels == 0
    ].sum()
# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    test_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "test_data.csv"
    )

    if not test_path.exists():

        raise FileNotFoundError(
            f"Test dataset not found at:\n{test_path}"
        )

    test_df = pd.read_csv(test_path)

    if "Date" in test_df.columns:

        test_df["Date"] = pd.to_datetime(
            test_df["Date"],
            format="%d-%b-%y",
            errors="coerce"
        )

    return test_df