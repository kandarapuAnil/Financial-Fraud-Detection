# ============================================================
# FINANCIAL FRAUD DETECTION DASHBOARD
# ============================================================
# Streamlit dashboard for the final Random Forest fraud model.
#
# Model:
#   RandomForestClassifier
#
# Dataset:
#   data/raw/synthetic_fraud_dataset_v2.csv
#
# Decision threshold:
#   Loaded from models/feature_information.pkl
#   Fallback: 0.17
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudGuard | Financial Fraud Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 15% 0%, rgba(99,102,241,.10), transparent 30%),
            radial-gradient(circle at 90% 10%, rgba(236,72,153,.08), transparent 28%),
            #0b1020;
    }

    [data-testid="stSidebar"] {
        background: #080d1a !important;
        border-right: 1px solid rgba(255,255,255,.06);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #151d35 0%, #10172b 55%, #171225 100%);
        border: 1px solid rgba(255,255,255,.07);
        box-shadow: 0 18px 55px rgba(0,0,0,.28);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.7px;
    }

    .hero p {
        color: #94a3b8;
        margin: .35rem 0 0;
        font-size: .92rem;
    }

    .badge {
        display: inline-block;
        margin-top: .9rem;
        padding: .38rem .85rem;
        border-radius: 999px;
        background: rgba(99,102,241,.13);
        color: #a5b4fc;
        border: 1px solid rgba(99,102,241,.25);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .5px;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 1.08rem;
        font-weight: 750;
        margin: 1.35rem 0 .8rem;
    }

    .metric-card {
        background: rgba(18,26,48,.86);
        border: 1px solid rgba(255,255,255,.065);
        border-radius: 15px;
        padding: 1.05rem 1.15rem;
        min-height: 118px;
        box-shadow: 0 8px 30px rgba(0,0,0,.16);
    }

    .metric-label {
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: .65rem;
        font-weight: 800;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: .3rem;
    }

    .metric-sub {
        color: #64748b;
        font-size: .72rem;
        margin-top: .2rem;
    }

    .risk-box {
        padding: 1.25rem 1.4rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(18,26,48,.85);
        margin: .5rem 0 1rem;
    }

    .risk-low {
        color: #4ade80;
        font-weight: 800;
    }

    .risk-medium {
        color: #fbbf24;
        font-weight: 800;
    }

    .risk-high {
        color: #fb7185;
        font-weight: 800;
    }

    .risk-critical {
        color: #f43f5e;
        font-weight: 900;
    }

    .result-fraud {
        padding: 1.4rem;
        border-radius: 16px;
        background: rgba(244,63,94,.09);
        border: 1px solid rgba(244,63,94,.25);
    }

    .result-safe {
        padding: 1.4rem;
        border-radius: 16px;
        background: rgba(34,197,94,.08);
        border: 1px solid rgba(34,197,94,.22);
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 850;
        color: #f8fafc;
    }

    .result-text {
        color: #94a3b8;
        font-size: .85rem;
        margin-top: .25rem;
    }

    .model-note {
        padding: 1rem 1.15rem;
        border-radius: 13px;
        background: rgba(255,255,255,.025);
        border: 1px solid rgba(255,255,255,.055);
        color: #94a3b8;
        font-size: .82rem;
        line-height: 1.6;
    }

    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 0;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 750;
        min-height: 2.65rem;
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #475569;
        font-size: .72rem;
        padding: 1.5rem 0 .4rem;
        border-top: 1px solid rgba(255,255,255,.05);
        margin-top: 2rem;
    }

    /* Make Streamlit inputs blend into the dashboard */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stTextInput"] > div,
    div[data-testid="stDateInput"] > div {
        background: rgba(255,255,255,.025);
        border-radius: 9px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "synthetic_fraud_dataset_v2.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
FEATURE_INFO_PATH = BASE_DIR / "models" / "feature_information.pkl"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_data.csv"


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_THRESHOLD = 0.17

NUMERIC_FEATURES = [
    "Transaction_Amount",
    "Account_Balance",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Card_Age",
    "Txn_to_Balance_Pct",
    "Amount_Zscore_User",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Hour",
]

CATEGORICAL_FEATURES = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type",
]


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
        )
    return joblib.load(PREPROCESSOR_PATH)


@st.cache_data
def load_test_data():
    if not TEST_DATA_PATH.exists():
        return None

    test = pd.read_csv(TEST_DATA_PATH)

    if "Date" in test.columns:
        test["Date"] = pd.to_datetime(
            test["Date"],
            errors="coerce"
        )

    return test


@st.cache_resource
def load_feature_information():
    if not FEATURE_INFO_PATH.exists():
        return {
            "fraud_threshold": DEFAULT_THRESHOLD
        }

    try:
        info = joblib.load(FEATURE_INFO_PATH)

        if not isinstance(info, dict):
            return {
                "fraud_threshold": DEFAULT_THRESHOLD
            }

        return info

    except Exception:
        return {
            "fraud_threshold": DEFAULT_THRESHOLD
        }


# ============================================================
# SAFE INITIALIZATION
# ============================================================

try:
    df = load_dataset()
    model = load_model()
    preprocessor = load_preprocessor()
    feature_info = load_feature_information()

except Exception as exc:
    st.error("Unable to initialize the Fraud Detection Dashboard.")
    st.exception(exc)
    st.stop()


FRAUD_THRESHOLD = float(
    feature_info.get(
        "fraud_threshold",
        DEFAULT_THRESHOLD
    )
)

if not 0 < FRAUD_THRESHOLD < 1:
    FRAUD_THRESHOLD = DEFAULT_THRESHOLD


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">
                ● MODEL ACTIVE &nbsp;|&nbsp; THRESHOLD {FRAUD_THRESHOLD:.2f}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_layout(fig, height=400):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#94a3b8",
            size=12
        ),
        title_font=dict(
            color="#e2e8f0",
            size=16
        ),
        margin=dict(
            t=55,
            b=40,
            l=25,
            r=25
        ),
        height=height,
    )

    return fig


def get_risk_level(probability):
    """
    Risk classification is intentionally separate from the
    fraud decision threshold.
    """

    if probability >= 0.40:
        return "CRITICAL", "risk-critical", "🔴"

    if probability >= FRAUD_THRESHOLD:
        return "HIGH", "risk-high", "🟠"

    if probability >= 0.10:
        return "MEDIUM", "risk-medium", "🟡"

    return "LOW", "risk-low", "🟢"


def prepare_prediction_input(values):
    """
    Recreates exactly the date-derived features used by
    train_final_model.py.

    Training transforms:
        Date -> Year, Month, Day, DayOfWeek, Hour
    """

    row = dict(values)

    transaction_date = pd.Timestamp(
        row.pop("Date")
    )

    row["Year"] = transaction_date.year
    row["Month"] = transaction_date.month
    row["Day"] = transaction_date.day
    row["DayOfWeek"] = transaction_date.dayofweek
    row["Hour"] = transaction_date.hour

    # Recalculate this deterministic engineered feature.
    balance = max(
        float(row["Account_Balance"]),
        1.0
    )

    row["Txn_to_Balance_Pct"] = (
        float(row["Transaction_Amount"])
        / balance
    ) * 100.0

    # Amount_Zscore_User cannot be reconstructed from a single
    # transaction after User_ID was intentionally excluded from
    # model training, so it remains an explicit model input.
    row["Amount_Zscore_User"] = float(
        row.get("Amount_Zscore_User", 0.0)
    )

    input_df = pd.DataFrame([row])

    # Use the exact fitted preprocessor feature order.
    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(
            preprocessor.feature_names_in_
        )

        missing = [
            col for col in expected
            if col not in input_df.columns
        ]

        if missing:
            raise ValueError(
                "The prediction input is missing model features: "
                + ", ".join(missing)
            )

        input_df = input_df[expected]

    # Force numeric values to real Python floats.
    for column in NUMERIC_FEATURES:
        if column in input_df.columns:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            ).astype(float)

    # Force categorical values to strings.
    for column in CATEGORICAL_FEATURES:
        if column in input_df.columns:
            input_df[column] = (
                input_df[column]
                .astype(str)
                .replace({
                    "nan": "Unknown",
                    "None": "Unknown",
                    "": "Unknown"
                })
            )

    # Final NaN protection.
    for column in input_df.columns:
        if column in NUMERIC_FEATURES:
            input_df[column] = input_df[column].fillna(0.0)
        else:
            input_df[column] = input_df[column].fillna("Unknown")

    return input_df


def predict_transaction(values):
    input_df = prepare_prediction_input(values)

    processed = preprocessor.transform(
        input_df
    )

    probabilities = model.predict_proba(
        processed
    )

    probability = float(
        np.asarray(probabilities)[0, 1]
    )

    prediction = int(
        probability >= FRAUD_THRESHOLD
    )

    return prediction, probability, input_df


def create_test_predictions():
    """
    Used only by Model Performance page.
    test_data.csv already contains the same engineered
    features used during training.
    """

    test_df = load_test_data()

    if test_df is None:
        return None

    if "Fraud_Label" not in test_df.columns:
        return None

    y_true = test_df["Fraud_Label"].astype(int)

    X = test_df.drop(
        columns=["Fraud_Label"],
        errors="ignore"
    ).copy()

    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(
            preprocessor.feature_names_in_
        )

        missing = [
            col for col in expected
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                "test_data.csv is missing model features: "
                + ", ".join(missing)
            )

        X = X[expected]

    for column in NUMERIC_FEATURES:
        if column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            ).fillna(0.0).astype(float)

    for column in CATEGORICAL_FEATURES:
        if column in X.columns:
            X[column] = (
                X[column]
                .astype(str)
                .replace({
                    "nan": "Unknown",
                    "None": "Unknown",
                    "": "Unknown"
                })
                .fillna("Unknown")
            )

    processed = preprocessor.transform(X)

    probability = model.predict_proba(
        processed
    )[:, 1]

    prediction = (
        probability >= FRAUD_THRESHOLD
    ).astype(int)

    return test_df, y_true, prediction, probability


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:1.1rem 0 1.3rem;
            border-bottom:1px solid rgba(255,255,255,.06);
            margin-bottom:1rem;
        ">
            <div style="font-size:2.5rem;">🚨</div>
            <div style="
                color:#f8fafc;
                font-size:1.25rem;
                font-weight:850;
            ">
                Fraud<span style="color:#818cf8;">Guard</span>
            </div>
            <div style="
                color:#64748b;
                font-size:.62rem;
                letter-spacing:2px;
                text-transform:uppercase;
                margin-top:.2rem;
            ">
                AI Risk Monitor
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Fraud Analysis",
            "Fraud Prediction",
            "Model Performance",
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        "<div style='margin-top:1.4rem;color:#64748b;"
        "font-size:.65rem;font-weight:800;letter-spacing:1px;"
        "text-transform:uppercase;'>System</div>",
        unsafe_allow_html=True
    )

    st.success("Model loaded")
    st.caption(
        f"Fraud threshold: {FRAUD_THRESHOLD:.2f}"
    )

    st.markdown(
        """
        <div style="
            margin-top:1rem;
            color:#64748b;
            font-size:.72rem;
            line-height:1.6;
        ">
        Random Forest<br>
        Supervised Classification<br>
        Synthetic Fraud Dataset
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET METRICS
# ============================================================

total_transactions = int(len(df))

fraud_transactions = int(
    (df["Fraud_Label"] == 1).sum()
)

legitimate_transactions = int(
    (df["Fraud_Label"] == 0).sum()
)

fraud_rate = (
    fraud_transactions
    / total_transactions
    * 100
    if total_transactions
    else 0
)

amount_series = pd.to_numeric(
    df["Transaction_Amount"],
    errors="coerce"
).fillna(0)

fraud_amount = float(
    amount_series[
        df["Fraud_Label"] == 1
    ].sum()
)

total_amount = float(
    amount_series.sum()
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    page_header(
        "🚨 Financial Fraud Detection",
        "Machine-learning based transaction monitoring and risk analytics."
    )

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Transactions",
            f"{total_transactions:,}",
            "Dataset volume"
        )

    with c2:
        metric_card(
            "Fraud Transactions",
            f"{fraud_transactions:,}",
            "Actual fraudulent records"
        )

    with c3:
        metric_card(
            "Fraud Rate",
            f"{fraud_rate:.2f}%",
            "Fraud / total transactions"
        )

    with c4:
        metric_card(
            "Fraud Amount",
            f"₹{fraud_amount:,.0f}",
            "Transaction amount in fraud records"
        )

    st.markdown(
        '<div class="section-title">Transaction Distribution</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        status_df = pd.DataFrame({
            "Status": [
                "Legitimate",
                "Fraud"
            ],
            "Transactions": [
                legitimate_transactions,
                fraud_transactions
            ]
        })

        fig = px.pie(
            status_df,
            names="Status",
            values="Transactions",
            hole=.55,
            title="Legitimate vs Fraudulent",
            color="Status",
            color_discrete_map={
                "Legitimate": "#4ade80",
                "Fraud": "#f43f5e"
            }
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            chart_layout(fig, 390),
            use_container_width=True
        )

    with col2:
        if "Transaction_Type" in df.columns:

            type_df = (
                df.groupby(
                    ["Transaction_Type", "Fraud_Label"]
                )
                .size()
                .reset_index(name="Count")
            )

            type_df["Status"] = type_df[
                "Fraud_Label"
            ].map({
                0: "Legitimate",
                1: "Fraud"
            })

            fig = px.bar(
                type_df,
                x="Transaction_Type",
                y="Count",
                color="Status",
                barmode="group",
                title="Transactions by Type",
                color_discrete_map={
                    "Legitimate": "#4ade80",
                    "Fraud": "#f43f5e"
                }
            )

            st.plotly_chart(
                chart_layout(fig, 390),
                use_container_width=True
            )

    st.markdown(
        '<div class="section-title">Fraud Activity Over Time</div>',
        unsafe_allow_html=True
    )

    if "Date" in df.columns:

        daily = (
            df.assign(
                DateOnly=df["Date"].dt.date
            )
            .groupby("DateOnly")["Fraud_Label"]
            .sum()
            .reset_index()
        )

        fig = px.area(
            daily,
            x="DateOnly",
            y="Fraud_Label",
            title="Daily Fraud Transactions",
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">System Risk Summary</div>',
        unsafe_allow_html=True
    )

    amount_fraud_share = (
        fraud_amount / total_amount * 100
        if total_amount
        else 0
    )

    if fraud_rate >= 10:
        overall_class = "risk-high"
        overall_text = "HIGH"
        overall_icon = "🔴"
    elif fraud_rate >= 5:
        overall_class = "risk-medium"
        overall_text = "MEDIUM"
        overall_icon = "🟡"
    else:
        overall_class = "risk-low"
        overall_text = "LOW"
        overall_icon = "🟢"

    st.markdown(
        f"""
        <div class="risk-box">
            <div style="font-size:.72rem;color:#64748b;
                        text-transform:uppercase;
                        letter-spacing:1px;font-weight:800;">
                Overall Dataset Risk
            </div>
            <div style="font-size:1.5rem;margin-top:.35rem;"
                 class="{overall_class}">
                {overall_icon} {overall_text}
            </div>
            <div style="color:#94a3b8;font-size:.82rem;margin-top:.45rem;">
                Fraud represents {fraud_rate:.2f}% of transactions and
                {amount_fraud_share:.2f}% of the total transaction amount.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2 — FRAUD ANALYSIS
# ============================================================

elif page == "Fraud Analysis":

    page_header(
        "🔎 Fraud Analysis",
        "Explore fraudulent activity across transaction and customer dimensions."
    )

    filtered = df.copy()

    st.sidebar.markdown(
        "### Analysis Filters"
    )

    if "Transaction_Type" in df.columns:
        transaction_types = [
            "All"
        ] + sorted(
            df["Transaction_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_type = st.sidebar.selectbox(
            "Transaction Type",
            transaction_types
        )

        if selected_type != "All":
            filtered = filtered[
                filtered["Transaction_Type"].astype(str)
                == selected_type
            ]

    if "Device_Type" in df.columns:
        devices = [
            "All"
        ] + sorted(
            df["Device_Type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_device = st.sidebar.selectbox(
            "Device",
            devices
        )

        if selected_device != "All":
            filtered = filtered[
                filtered["Device_Type"].astype(str)
                == selected_device
            ]

    if "Location" in df.columns:
        locations = [
            "All"
        ] + sorted(
            df["Location"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_location = st.sidebar.selectbox(
            "Location",
            locations
        )

        if selected_location != "All":
            filtered = filtered[
                filtered["Location"].astype(str)
                == selected_location
            ]

    if "Merchant_Category" in df.columns:
        merchants = [
            "All"
        ] + sorted(
            df["Merchant_Category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_merchant = st.sidebar.selectbox(
            "Merchant Category",
            merchants
        )

        if selected_merchant != "All":
            filtered = filtered[
                filtered["Merchant_Category"].astype(str)
                == selected_merchant
            ]

    if "Date" in df.columns:

        valid_dates = df["Date"].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            selected_dates = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if (
                isinstance(selected_dates, tuple)
                and len(selected_dates) == 2
            ):
                start, end = selected_dates

                filtered = filtered[
                    (
                        filtered["Date"].dt.date
                        >= start
                    )
                    & (
                        filtered["Date"].dt.date
                        <= end
                    )
                ]

    if filtered.empty:
        st.warning(
            "No transactions match the selected filters."
        )
        st.stop()

    filtered_fraud = int(
        (filtered["Fraud_Label"] == 1).sum()
    )

    filtered_rate = (
        filtered_fraud
        / len(filtered)
        * 100
    )

    filtered_amount = float(
        pd.to_numeric(
            filtered["Transaction_Amount"],
            errors="coerce"
        )
        .fillna(0)
        .loc[
            filtered["Fraud_Label"] == 1
        ]
        .sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Filtered Transactions",
            f"{len(filtered):,}"
        )

    with c2:
        metric_card(
            "Fraud Transactions",
            f"{filtered_fraud:,}"
        )

    with c3:
        metric_card(
            "Fraud Rate",
            f"{filtered_rate:.2f}%"
        )

    with c4:
        metric_card(
            "Fraud Amount",
            f"₹{filtered_amount:,.0f}"
        )

    st.markdown(
        '<div class="section-title">Fraud by Location</div>',
        unsafe_allow_html=True
    )

    if "Location" in filtered.columns:

        location_df = (
            filtered[filtered["Fraud_Label"] == 1]
            .groupby("Location")
            .size()
            .reset_index(name="Fraud Count")
            .sort_values(
                "Fraud Count",
                ascending=False
            )
        )

        fig = px.bar(
            location_df,
            x="Location",
            y="Fraud Count",
            color="Fraud Count",
            color_continuous_scale="Reds",
            text_auto=True
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        if "Merchant_Category" in filtered.columns:

            merchant_df = (
                filtered[filtered["Fraud_Label"] == 1]
                .groupby("Merchant_Category")
                .size()
                .reset_index(name="Fraud Count")
                .sort_values(
                    "Fraud Count",
                    ascending=False
                )
            )

            fig = px.bar(
                merchant_df,
                x="Merchant_Category",
                y="Fraud Count",
                color="Fraud Count",
                color_continuous_scale="Reds",
                text_auto=True,
                title="Fraud by Merchant Category"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

    with col2:

        if "Device_Type" in filtered.columns:

            device_df = (
                filtered[filtered["Fraud_Label"] == 1]
                .groupby("Device_Type")
                .size()
                .reset_index(name="Fraud Count")
            )

            fig = px.pie(
                device_df,
                names="Device_Type",
                values="Fraud Count",
                hole=.5,
                title="Fraud by Device Type"
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True
            )

    st.markdown(
        '<div class="section-title">Fraud Heatmap</div>',
        unsafe_allow_html=True
    )

    if (
        "Transaction_Type" in filtered.columns
        and "Device_Type" in filtered.columns
    ):

        heatmap = pd.crosstab(
            filtered["Transaction_Type"],
            filtered["Device_Type"],
            values=filtered["Fraud_Label"],
            aggfunc="sum"
        ).fillna(0)

        fig = px.imshow(
            heatmap,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(
            chart_layout(fig),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Fraudulent Transactions</div>',
        unsafe_allow_html=True
    )

    fraud_table = (
        filtered[
            filtered["Fraud_Label"] == 1
        ]
        .sort_values(
            "Transaction_Amount",
            ascending=False
        )
        .head(100)
    )

    st.dataframe(
        fraud_table,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Fraud Report",
        fraud_table.to_csv(index=False).encode("utf-8"),
        "filtered_fraud_transactions.csv",
        "text/csv",
        use_container_width=True
    )


# ============================================================
# PAGE 3 — FRAUD PREDICTION
# ============================================================

elif page == "Fraud Prediction":

    page_header(
        "🤖 Fraud Prediction",
        "Enter transaction information and estimate its fraud probability."
    )

    st.markdown(
        '<div class="section-title">Transaction Details</div>',
        unsafe_allow_html=True
    )

    with st.form("fraud_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.01,
                value=100.0,
                step=10.0
            )

            account_balance = st.number_input(
                "Account Balance",
                min_value=0.01,
                value=50000.0,
                step=500.0
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                [
                    "POS",
                    "Online",
                    "ATM Withdrawal",
                    "Bank Transfer"
                ]
            )

            device_type = st.selectbox(
                "Device Type",
                [
                    "Mobile",
                    "Laptop",
                    "Tablet"
                ]
            )

            location = st.selectbox(
                "Location",
                [
                    "Mumbai",
                    "New York",
                    "London",
                    "Tokyo",
                    "Sydney"
                ]
            )

            merchant_category = st.selectbox(
                "Merchant Category",
                [
                    "Groceries",
                    "Electronics",
                    "Travel",
                    "Restaurants",
                    "Clothing"
                ]
            )

        with col2:

            previous_fraud = st.selectbox(
                "Previous Fraudulent Activity",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

            daily_count = st.number_input(
                "Daily Transaction Count",
                min_value=1,
                max_value=100,
                value=7,
                step=1
            )

            card_type = st.selectbox(
                "Card Type",
                [
                    "Visa",
                    "Mastercard",
                    "Amex",
                    "Discover"
                ]
            )

            card_age = st.number_input(
                "Card Age (days)",
                min_value=1,
                max_value=5000,
                value=120,
                step=1
            )

            transaction_date = st.date_input(
                "Transaction Date",
                value=pd.Timestamp(
                    "2025-06-15"
                ).date()
            )

            transaction_time = st.time_input(
                "Transaction Time",
                value=pd.Timestamp(
                    "2025-06-15 14:30"
                ).time()
            )

            amount_zscore = st.number_input(
                "Amount Z-score for User",
                value=0.0,
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                help=(
                    "This feature is supplied explicitly because "
                    "User_ID is excluded from model training. "
                    "Use 0 for a typical transaction."
                )
            )

        st.markdown(
            """
            <div class="model-note">
                <strong>How the model works:</strong>
                The transaction is transformed using the same fitted
                preprocessor used during training. The Random Forest then
                returns a fraud probability. A probability of
                <strong>17% or higher</strong> is classified as fraud.
            </div>
            """,
            unsafe_allow_html=True
        )

        submitted = st.form_submit_button(
            "🔍 Analyze Transaction Risk",
            use_container_width=True
        )

    if submitted:

        transaction_datetime = pd.Timestamp(
            f"{transaction_date} {transaction_time}"
        )

        values = {
            "Transaction_Amount": float(
                transaction_amount
            ),
            "Transaction_Type": str(
                transaction_type
            ),
            "Account_Balance": float(
                account_balance
            ),
            "Device_Type": str(
                device_type
            ),
            "Location": str(
                location
            ),
            "Merchant_Category": str(
                merchant_category
            ),
            "Previous_Fraudulent_Activity": int(
                previous_fraud
            ),
            "Daily_Transaction_Count": int(
                daily_count
            ),
            "Card_Type": str(
                card_type
            ),
            "Card_Age": float(
                card_age
            ),
            "Date": transaction_datetime,
            "Amount_Zscore_User": float(
                amount_zscore
            ),
        }

        try:

            prediction, probability, _ = (
                predict_transaction(values)
            )

            risk, risk_class, risk_icon = (
                get_risk_level(probability)
            )

            probability_percent = (
                probability * 100
            )

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result-fraud">
                        <div class="result-title">
                            🚨 FRAUD DETECTED
                        </div>
                        <div class="result-text">
                            This transaction crossed the configured
                            fraud decision threshold of
                            {FRAUD_THRESHOLD:.2f}.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-safe">
                        <div class="result-title">
                            ✅ TRANSACTION APPEARS LEGITIMATE
                        </div>
                        <div class="result-text">
                            The predicted probability is below the
                            fraud decision threshold of
                            {FRAUD_THRESHOLD:.2f}.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "Fraud Probability",
                    f"{probability_percent:.2f}%",
                    "Model probability"
                )

            with c2:
                metric_card(
                    "Decision Threshold",
                    f"{FRAUD_THRESHOLD * 100:.0f}%",
                    "Optimized threshold"
                )

            with c3:
                metric_card(
                    "Risk Level",
                    f"{risk_icon} {risk}",
                    "Transaction risk"
                )

            st.progress(
                float(
                    np.clip(
                        probability,
                        0.0,
                        1.0
                    )
                )
            )

            if risk == "CRITICAL":
                st.error(
                    "Critical risk: immediate manual verification is recommended."
                )

            elif risk == "HIGH":
                st.warning(
                    "High risk: additional transaction verification is recommended."
                )

            elif risk == "MEDIUM":
                st.warning(
                    "Medium risk: monitor the transaction and consider additional checks."
                )

            else:
                st.success(
                    "Low risk: no immediate fraud action is indicated by the model."
                )

            st.markdown(
                '<div class="section-title">Transaction Summary</div>',
                unsafe_allow_html=True
            )

            summary = pd.DataFrame({
                "Field": [
                    "Transaction Amount",
                    "Account Balance",
                    "Transaction Type",
                    "Device",
                    "Location",
                    "Merchant Category",
                    "Previous Fraud Activity",
                    "Daily Transaction Count",
                    "Card Type",
                    "Card Age",
                    "Transaction Date/Time",
                ],
                "Value": [
                    f"₹{transaction_amount:,.2f}",
                    f"₹{account_balance:,.2f}",
                    transaction_type,
                    device_type,
                    location,
                    merchant_category,
                    "Yes" if previous_fraud else "No",
                    str(daily_count),
                    card_type,
                    f"{card_age:.0f} days",
                    str(transaction_datetime)
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as exc:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(exc)

            st.info(
                "If this error mentions feature names or preprocessing, "
                "re-run src/train_final_model.py so the model, preprocessor "
                "and feature information are generated together."
            )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    page_header(
        "📊 Model Performance",
        "Evaluate the final Random Forest using the saved test dataset."
    )

    try:

        performance = create_test_predictions()

    except Exception as exc:

        st.error(
            "Unable to evaluate the saved model."
        )

        st.exception(exc)
        st.stop()

    if performance is None:

        st.warning(
            "Test data was not found. Run src/train_final_model.py first."
        )

        st.stop()

    test_df, y_test, y_pred, y_probability = performance

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

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card(
            "Accuracy",
            f"{accuracy * 100:.2f}%",
            "Thresholded predictions"
        )

    with c2:
        metric_card(
            "Precision",
            f"{precision * 100:.2f}%",
            "Fraud precision"
        )

    with c3:
        metric_card(
            "Recall",
            f"{recall * 100:.2f}%",
            "Fraud detection rate"
        )

    with c4:
        metric_card(
            "F1 Score",
            f"{f1 * 100:.2f}%",
            "Precision / recall balance"
        )

    with c5:
        metric_card(
            "ROC-AUC",
            f"{auc:.4f}",
            "Probability ranking quality"
        )

    st.markdown(
        '<div class="section-title">Selected Fraud Threshold</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="model-note">
            <strong>Decision threshold: {FRAUD_THRESHOLD:.2f}</strong><br>
            The default Random Forest threshold of 0.50 was not used.
            Threshold optimization selected 0.17 because it produced the
            best F1-score during the dedicated threshold analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    labels = [
        "Legitimate",
        "Fraud"
    ]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        aspect="auto",
        color_continuous_scale="Reds",
        labels={
            "x": "Predicted",
            "y": "Actual"
        }
    )

    st.plotly_chart(
        chart_layout(fig, 420),
        use_container_width=True
    )

    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "True Negatives",
                f"{tn:,}",
                "Correct legitimate decisions"
            )

        with c2:
            metric_card(
                "False Positives",
                f"{fp:,}",
                "Legitimate flagged as fraud"
            )

        with c3:
            metric_card(
                "False Negatives",
                f"{fn:,}",
                "Fraud missed"
            )

        with c4:
            metric_card(
                "True Positives",
                f"{tp:,}",
                "Fraud correctly detected"
            )

    st.markdown(
        '<div class="section-title">ROC Curve</div>',
        unsafe_allow_html=True
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr
    })

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve — AUC {auc:.4f}"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(
            dash="dash",
            color="#64748b"
        )
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Precision-Recall Curve</div>',
        unsafe_allow_html=True
    )

    pr_precision, pr_recall, _ = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )

    pr_df = pd.DataFrame({
        "Recall": pr_recall,
        "Precision": pr_precision
    })

    fig = px.line(
        pr_df,
        x="Recall",
        y="Precision",
        title="Precision-Recall Curve"
    )

    fig.add_vline(
        x=recall,
        line_dash="dash",
        line_color="#f43f5e",
        annotation_text=f"Recall {recall:.2f}"
    )

    fig.add_hline(
        y=precision,
        line_dash="dash",
        line_color="#818cf8",
        annotation_text=f"Precision {precision:.2f}"
    )

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Model Interpretation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="model-note">
            The final Random Forest achieved an ROC-AUC of
            <strong>{auc:.4f}</strong>. Because fraud detection is an
            imbalanced classification problem, the dashboard reports
            precision, recall and F1-score in addition to accuracy.
            The operational fraud threshold is
            <strong>{FRAUD_THRESHOLD:.2f}</strong>, selected through
            threshold optimization rather than relying on the default
            0.50 cutoff.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚨 <strong>FraudGuard</strong> · Financial Fraud Detection
        · Machine Learning · Data Analytics · Streamlit
        <br>
        Financial Fraud Detection Project Developed by Anil
    </div>
    """,
    unsafe_allow_html=True
)