from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ROWS = 50000

rng = np.random.default_rng(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_fraud_dataset_v2.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BASIC TRANSACTION INFORMATION
# ============================================================

transaction_id = np.arange(
    1000001,
    1000001 + N_ROWS
)

user_id = rng.integers(
    10001,
    20001,
    size=N_ROWS
)


transaction_amount = np.round(
    rng.lognormal(
        mean=4.0,
        sigma=0.8,
        size=N_ROWS
    ),
    2
)

transaction_amount = np.clip(
    transaction_amount,
    1,
    5000
)


# ============================================================
# CATEGORICAL VARIABLES
# ============================================================

transaction_type = rng.choice(
    [
        "POS",
        "Online",
        "ATM Withdrawal",
        "Bank Transfer"
    ],
    size=N_ROWS,
    p=[
        0.25,
        0.30,
        0.20,
        0.25
    ]
)


device_type = rng.choice(
    [
        "Mobile",
        "Laptop",
        "Tablet"
    ],
    size=N_ROWS,
    p=[
        0.55,
        0.30,
        0.15
    ]
)


location = rng.choice(
    [
        "Mumbai",
        "New York",
        "London",
        "Tokyo",
        "Sydney"
    ],
    size=N_ROWS,
    p=[
        0.30,
        0.20,
        0.20,
        0.15,
        0.15
    ]
)


merchant_category = rng.choice(
    [
        "Groceries",
        "Electronics",
        "Travel",
        "Restaurants",
        "Clothing"
    ],
    size=N_ROWS
)


card_type = rng.choice(
    [
        "Visa",
        "Mastercard",
        "Amex",
        "Discover"
    ],
    size=N_ROWS
)


# ============================================================
# NUMERICAL FEATURES
# ============================================================

account_balance = np.round(
    rng.uniform(
        500,
        100000,
        size=N_ROWS
    ),
    2
)


previous_fraudulent_activity = rng.binomial(
    1,
    0.10,
    size=N_ROWS
)


daily_transaction_count = rng.poisson(
    7,
    size=N_ROWS
) + 1


daily_transaction_count = np.clip(
    daily_transaction_count,
    1,
    30
)


card_age = rng.integers(
    1,
    240,
    size=N_ROWS
)


# ============================================================
# DATE
# ============================================================

dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31",
    periods=N_ROWS
)

dates = pd.Series(
    dates
)

hour = dates.dt.hour.to_numpy()


# ============================================================
# DERIVED FEATURES
# ============================================================

txn_to_balance_pct = (
    transaction_amount
    / np.maximum(
        account_balance,
        1
    )
) * 100


user_mean_amount = (
    pd.Series(transaction_amount)
    .groupby(user_id)
    .transform("mean")
    .to_numpy()
)


user_std_amount = (
    pd.Series(transaction_amount)
    .groupby(user_id)
    .transform("std")
    .fillna(1)
    .to_numpy()
)


amount_zscore_user = (
    transaction_amount
    - user_mean_amount
) / np.maximum(
    user_std_amount,
    1
)


# ============================================================
# FRAUD RISK SCORE
# ============================================================

risk_score = np.zeros(
    N_ROWS,
    dtype=float
)


# Large transaction

risk_score += (
    transaction_amount > 800
) * 1.5


risk_score += (
    transaction_amount > 1500
) * 1.5


# High transaction-to-balance ratio

risk_score += (
    txn_to_balance_pct > 20
) * 1.5


risk_score += (
    txn_to_balance_pct > 50
) * 2.0


# Previous fraud history

risk_score += (
    previous_fraudulent_activity == 1
) * 2.5


# Unusually high transaction frequency

risk_score += (
    daily_transaction_count > 15
) * 1.5


risk_score += (
    daily_transaction_count > 22
) * 2.0


# Unusual transaction amount

risk_score += (
    amount_zscore_user > 2
) * 2.0


risk_score += (
    amount_zscore_user > 3
) * 2.0


# Higher-risk transaction channels

risk_score += (
    transaction_type == "Online"
) * 0.5


risk_score += (
    transaction_type == "Bank Transfer"
) * 0.4


# Certain suspicious device behavior

risk_score += (
    device_type == "Tablet"
) * 0.3


# ============================================================
# RANDOM NOISE
# ============================================================

risk_score += rng.normal(
    0,
    0.35,
    N_ROWS
)


# ============================================================
# CONVERT RISK SCORE TO FRAUD PROBABILITY
# ============================================================

fraud_probability = (
    1
    /
    (
        1
        +
        np.exp(
            -(
                risk_score
                - 3.5
            )
        )
    )
)


# ============================================================
# FRAUD LABEL
# ============================================================

fraud_label = (
    rng.random(
        N_ROWS
    )
    <
    fraud_probability
).astype(int)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    {
        "Transaction_ID": transaction_id,
        "User_ID": user_id,
        "Transaction_Amount": transaction_amount,
        "Transaction_Type": transaction_type,
        "Date": dates,
        "Account_Balance": account_balance,
        "Device_Type": device_type,
        "Location": location,
        "Merchant_Category": merchant_category,
        "Previous_Fraudulent_Activity": previous_fraudulent_activity,
        "Daily_Transaction_Count": daily_transaction_count,
        "Card_Type": card_type,
        "Card_Age": card_age,
        "Fraud_Label": fraud_label,
        "Txn_to_Balance_Pct": np.round(
            txn_to_balance_pct,
            4
        ),
        "Amount_Zscore_User": np.round(
            amount_zscore_user,
            4
        )
    }
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("NEW SYNTHETIC FRAUD DATASET CREATED")
print("=" * 70)

print(
    f"\nRows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"\nFraud transactions:"
)

print(
    df["Fraud_Label"].value_counts()
)

print(
    "\nFraud percentage:"
)

print(
    df["Fraud_Label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\nDataset preview:"
)

print(
    df.head()
)