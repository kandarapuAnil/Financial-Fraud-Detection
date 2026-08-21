from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
TEST_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FRAUD DETECTION THRESHOLD OPTIMIZATION")
print("=" * 70)

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

test_df = pd.read_csv(TEST_PATH)

print("\nTest data shape:")
print(test_df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "Fraud_Label"

y_test = test_df[TARGET].astype(int)


# ============================================================
# DROP TARGET
# ============================================================

X_test = test_df.drop(
    columns=[TARGET]
)


# ============================================================
# TRANSFORM
# ============================================================

X_test_transformed = preprocessor.transform(
    X_test
)


# ============================================================
# FRAUD PROBABILITY
# ============================================================

fraud_probability = model.predict_proba(
    X_test_transformed
)[:, 1]


# ============================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.05,
    0.51,
    0.01
)


results = []


for threshold in thresholds:

    y_pred = (
        fraud_probability >= threshold
    ).astype(int)

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

    results.append(
        {
            "Threshold": round(
                float(threshold),
                2
            ),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Fraud_Predictions": int(
                y_pred.sum()
            )
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["F1"].idxmax()
]


# ============================================================
# BEST RECALL WITH PRECISION >= 20%
# ============================================================

acceptable_precision = results_df[
    results_df["Precision"] >= 0.20
]

if not acceptable_precision.empty:

    best_recall = acceptable_precision.loc[
        acceptable_precision["Recall"].idxmax()
    ]

else:

    best_recall = None


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

display_df = results_df.copy()

display_df["Precision"] = (
    display_df["Precision"] * 100
).round(2)

display_df["Recall"] = (
    display_df["Recall"] * 100
).round(2)

display_df["F1"] = (
    display_df["F1"] * 100
).round(2)

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BEST F1
# ============================================================

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_f1['Threshold']:.2f}"
)

print(
    f"Precision : {best_f1['Precision']:.4f}"
)

print(
    f"Recall    : {best_f1['Recall']:.4f}"
)

print(
    f"F1 Score  : {best_f1['F1']:.4f}"
)

print(
    f"Fraud Predictions : "
    f"{int(best_f1['Fraud_Predictions'])}"
)


# ============================================================
# BEST RECALL
# ============================================================

if best_recall is not None:

    print("\n" + "=" * 70)
    print("BEST RECALL WITH PRECISION >= 20%")
    print("=" * 70)

    print(
        f"Threshold : {best_recall['Threshold']:.2f}"
    )

    print(
        f"Precision : {best_recall['Precision']:.4f}"
    )

    print(
        f"Recall    : {best_recall['Recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_recall['F1']:.4f}"
    )

    print(
        f"Fraud Predictions : "
        f"{int(best_recall['Fraud_Predictions'])}"
    )


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

y_default = (
    fraud_probability >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("DEFAULT 0.50 THRESHOLD")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_default,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_default
    )
)


print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 70)