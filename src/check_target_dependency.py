from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_fraud_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

TARGET = "Fraud_Label"


print("=" * 70)
print("TARGET DEPENDENCY ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

y = df[TARGET].astype(int)


# ------------------------------------------------------------
# REMOVE IDENTIFIERS
# ------------------------------------------------------------

drop_columns = [
    TARGET,
    "Transaction_ID",
    "User_ID",
    "Date"
]

X = df.drop(
    columns=[
        column
        for column in drop_columns
        if column in df.columns
    ]
).copy()


# ------------------------------------------------------------
# HANDLE CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()


if categorical_columns:

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X[categorical_columns] = encoder.fit_transform(
        X[categorical_columns].astype(str)
    )


# ------------------------------------------------------------
# CLEAN NUMERIC DATA
# ------------------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


# ------------------------------------------------------------
# MUTUAL INFORMATION
# ------------------------------------------------------------

print("\nCalculating mutual information...\n")

mi = mutual_info_classif(
    X,
    y,
    random_state=42
)


mi_results = pd.DataFrame(
    {
        "Feature": X.columns,
        "Mutual_Information": mi
    }
).sort_values(
    "Mutual_Information",
    ascending=False
)


print(
    mi_results.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# TARGET SHUFFLE TEST
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TARGET SHUFFLE TEST")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


probabilities = model.predict_proba(
    X_test
)[:, 1
]

auc_original = roc_auc_score(
    y_test,
    probabilities
)


print(
    f"\nOriginal target ROC-AUC: {auc_original:.4f}"
)


# ------------------------------------------------------------
# SHUFFLED TARGET
# ------------------------------------------------------------

y_shuffled = y.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_shuffled,
    test_size=0.20,
    random_state=42,
    stratify=y_shuffled
)


model_shuffled = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model_shuffled.fit(
    X_train,
    y_train
)


probabilities_shuffled = (
    model_shuffled
    .predict_proba(X_test)[:, 1]
)


auc_shuffled = roc_auc_score(
    y_test,
    probabilities_shuffled
)


print(
    f"Shuffled target ROC-AUC: {auc_shuffled:.4f}"
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)