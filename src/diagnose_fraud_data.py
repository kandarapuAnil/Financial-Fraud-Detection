from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_fraud_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FRAUD DATASET DIAGNOSTIC")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# TARGET
# ============================================================

TARGET = "Fraud_Label"

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget percentage:")
print(
    df[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(3)
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(
    missing[missing > 0]
)

if missing.sum() == 0:
    print("No missing values found.")


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate rows:")

print(
    df.duplicated().sum()
)


# ============================================================
# FRAUD RATE BY CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type"
]

print("\n" + "=" * 70)
print("FRAUD RATE BY CATEGORICAL FEATURES")
print("=" * 70)

for column in categorical_columns:

    if column not in df.columns:
        continue

    print("\n")
    print("-" * 60)
    print(column)
    print("-" * 60)

    result = (
        df.groupby(column)[TARGET]
        .agg(
            transactions="count",
            fraud_count="sum",
            fraud_rate="mean"
        )
        .sort_values(
            "fraud_rate",
            ascending=False
        )
    )

    result["fraud_rate"] = (
        result["fraud_rate"] * 100
    ).round(2)

    print(result)


# ============================================================
# NUMERICAL FEATURE ANALYSIS
# ============================================================

numeric_columns = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Txn_to_Balance_Pct",
    "Amount_Zscore_User"
]

print("\n" + "=" * 70)
print("NUMERICAL FEATURE ANALYSIS")
print("=" * 70)

for column in numeric_columns:

    if column not in df.columns:
        continue

    print("\n")
    print("-" * 60)
    print(column)
    print("-" * 60)

    stats = (
        df.groupby(TARGET)[column]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(
        stats.round(4)
    )


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL CORRELATION WITH FRAUD LABEL")
print("=" * 70)

numeric_existing = [
    column
    for column in numeric_columns
    if column in df.columns
]

correlations = (
    df[
        numeric_existing + [TARGET]
    ]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(
    correlations.round(4)
)


# ============================================================
# CHECK FRAUD LABEL RANDOMNESS
# ============================================================

print("\n" + "=" * 70)
print("FRAUD LABEL CHECK")
print("=" * 70)

print(
    "\nFraud ratio:"
)

print(
    df[TARGET].mean()
)


# ============================================================
# SIMPLE RULE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SIMPLE FRAUD PATTERN CHECKS")
print("=" * 70)


if "Previous_Fraudulent_Activity" in df.columns:

    print(
        "\nFraud rate by Previous_Fraudulent_Activity:"
    )

    print(
        df.groupby(
            "Previous_Fraudulent_Activity"
        )[TARGET]
        .agg(
            ["count", "mean"]
        )
        .assign(
            fraud_rate=lambda x:
            x["mean"] * 100
        )
        .drop(
            columns=["mean"]
        )
        .round(2)
    )


if "Txn_to_Balance_Pct" in df.columns:

    print(
        "\nTransaction-to-balance percentage by fraud:"
    )

    print(
        df.groupby(TARGET)[
            "Txn_to_Balance_Pct"
        ]
        .agg(
            [
                "mean",
                "median",
                "std"
            ]
        )
        .round(4)
    )


if "Amount_Zscore_User" in df.columns:

    print(
        "\nAmount Z-score by fraud:"
    )

    print(
        df.groupby(TARGET)[
            "Amount_Zscore_User"
        ]
        .agg(
            [
                "mean",
                "median",
                "std"
            ]
        )
        .round(4)
    )


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)