# ============================================================
# FINAL FRAUD DETECTION MODEL TRAINING
# ============================================================

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "synthetic_fraud_dataset_v2.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "fraud_model.pkl"
)

PREPROCESSOR_PATH = (
    MODEL_DIR
    / "preprocessor.pkl"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_data.csv"
)

TEST_DATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINANCIAL FRAUD DETECTION - FINAL MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)

print(
    "\nColumns:"
)

for column in df.columns:
    print(
        f"  - {column}"
    )


# ============================================================
# TARGET
# ============================================================

TARGET = "Fraud_Label"

# Probability threshold selected from threshold optimization.
# At 0.17, the previous evaluation achieved the best F1 score
# among the tested thresholds (approximately 40.38%).
FRAUD_THRESHOLD = 0.17

if TARGET not in df.columns:

    raise ValueError(
        f"""
Target column '{TARGET}' was not found.

Available columns:

{list(df.columns)}
"""
    )


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.copy()

# Remove duplicate rows

df = df.drop_duplicates()

print(
    f"\nShape after removing duplicates: {df.shape}"
)


# ============================================================
# REMOVE IDENTIFIER COLUMNS
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "Transaction_ID",
    "User_ID"
]

feature_columns = [
    column
    for column in df.columns
    if column not in DROP_COLUMNS
]


X = df[feature_columns].copy()

y = df[TARGET].copy()


# ============================================================
# DATE HANDLING
# ============================================================

if "Date" in X.columns:

    X["Date"] = pd.to_datetime(
        X["Date"],
        errors="coerce"
    )

    # Convert date into useful numeric features

    X["Year"] = (
        X["Date"]
        .dt.year
    )

    X["Month"] = (
        X["Date"]
        .dt.month
    )

    X["Day"] = (
        X["Date"]
        .dt.day
    )

    X["DayOfWeek"] = (
        X["Date"]
        .dt.dayofweek
    )

    X["Hour"] = (
        X["Date"]
        .dt.hour
    )

    X = X.drop(
        columns=["Date"]
    )


# ============================================================
# CONVERT TARGET
# ============================================================

y = pd.to_numeric(
    y,
    errors="coerce"
)

valid_rows = y.notna()

X = X.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].astype(int)


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nFraud distribution:")

print(
    y.value_counts()
)

print("\nFraud percentage:")

print(
    y.value_counts(
        normalize=True
    ) * 100
)


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "int32",
        "int16",
        "int8",
        "float64",
        "float32"
    ]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumeric features:")

for column in numeric_features:
    print(
        f"  - {column}"
    )


print("\nCategorical features:")

for column in categorical_features:
    print(
        f"  - {column}"
    )


# ============================================================
# NUMERIC PIPELINE
# ============================================================

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


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

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


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    f"\nTraining samples: {len(X_train):,}"
)

print(
    f"Testing samples: {len(X_test):,}"
)


# ============================================================
# TRANSFORM TRAINING DATA
# ============================================================

print(
    "\nFitting preprocessor..."
)

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)


X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nTraining Random Forest..."
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

# Generate fraud probabilities first.
y_probability = model.predict_proba(
    X_test_processed
)[:, 1]

# Do NOT use RandomForest's default 0.50 threshold.
# Fraud detection is recall-sensitive, so we use the
# threshold selected during threshold optimization.
y_pred = (
    y_probability >= FRAUD_THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"Fraud Threshold : {FRAUD_THRESHOLD:.2f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving model..."
)

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

print(
    "Saving preprocessor..."
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)


# ============================================================
# SAVE TEST DATA
# ============================================================

test_data = X_test.copy()

test_data[TARGET] = y_test.values

test_data.to_csv(
    TEST_DATA_PATH,
    index=False
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_information = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_features": feature_columns,
    "target": TARGET,
    "fraud_threshold": FRAUD_THRESHOLD,
    "model_type": "RandomForestClassifier",
    "threshold_selection": "Best F1 from threshold optimization"
}

FEATURE_INFO_PATH = (
    MODEL_DIR
    / "feature_information.pkl"
)

joblib.dump(
    feature_information,
    FEATURE_INFO_PATH
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nModel saved to:"
)

print(
    MODEL_PATH
)

print(
    f"\nPreprocessor saved to:"
)

print(
    PREPROCESSOR_PATH
)

print(
    f"\nTest data saved to:"
)

print(
    TEST_DATA_PATH
)

print(
    f"\nFeature information saved to:"
)

print(
    FEATURE_INFO_PATH
)

print(
    "\nYou can now run:"
)

print(
    "streamlit run dashboard/app.py"
)

print(
    f"\nFraud decision threshold: {FRAUD_THRESHOLD:.2f}"
)