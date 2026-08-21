FINANCIAL FRAUD DETECTION SYSTEM
================================

PROJECT OVERVIEW
================

The Financial Fraud Detection System is a Machine Learning and Data Analytics project designed to identify potentially fraudulent financial transactions and provide actionable insights through an interactive Streamlit dashboard.

The project combines:

- Data Exploration
- Data Diagnostics
- Feature Engineering
- Data Preprocessing
- Machine Learning
- Fraud Classification
- Threshold Optimization
- Model Evaluation
- Interactive Data Visualization
- Fraud Risk Prediction

The final system uses a Random Forest Classification Model with an optimized fraud decision threshold of 0.17.


PROJECT OBJECTIVES
==================

The main objectives of this project are:

- Detect potentially fraudulent financial transactions.
- Analyze transaction patterns and fraud distribution.
- Perform exploratory data analysis.
- Perform dataset diagnostics.
- Engineer useful transaction features.
- Train and evaluate a Machine Learning model.
- Optimize the fraud classification threshold.
- Provide an interactive Streamlit dashboard.
- Display fraud-risk predictions.
- Present model performance using multiple evaluation metrics.


PROJECT WORKFLOW
================

Raw Dataset
     |
     v
Data Exploration
     |
     v
Data Diagnostics
     |
     v
Feature Engineering
     |
     v
Data Preprocessing
     |
     v
Random Forest Model Training
     |
     v
Threshold Optimization
     |
     v
Model Evaluation
     |
     v
Streamlit Dashboard


PROJECT STRUCTURE
=================

Financial_Fraud_Detection/

├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   └── utils.py
│
├── data/
│   ├── raw/
│   │   └── synthetic_fraud_dataset_v2.csv
│   │
│   └── processed/
│       ├── X_train_processed.npz
│       ├── X_test_processed.npz
│       └── test_data.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_information.pkl
│   └── final_model_name.txt
│
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── reports/
│   ├── feature_importance.csv
│   ├── final_model_evaluation.csv
│   ├── model_comparison.csv
│   └── model_predictions.csv
│
├── src/
│   ├── generate_fraud_dataset.py
│   ├── diagnose_fraud_data.py
│   ├── check_target_dependency.py
│   ├── optimize_threshold.py
│   └── train_final_model.py
│
├── .gitignore
├── requirements.txt
└── README.txt


DATASET
=======

The dataset contains approximately 50,000 financial transactions and includes transaction, account, device, location, merchant, card, and fraud-related attributes.

DATASET FEATURES
================

Transaction_ID
Unique transaction identifier.

User_ID
Unique user identifier.

Transaction_Amount
Amount involved in the transaction.

Transaction_Type
Type of financial transaction.

Date
Transaction date and time.

Account_Balance
Available account balance.

Device_Type
Device used to perform the transaction.

Location
Transaction location.

Merchant_Category
Category of the merchant.

Previous_Fraudulent_Activity
Indicates previous fraudulent activity.

Daily_Transaction_Count
Number of transactions performed during the day.

Card_Type
Type of payment card.

Card_Age
Age of the card.

Fraud_Label
Target variable.

0 = Legitimate Transaction
1 = Fraudulent Transaction

Txn_to_Balance_Pct
Transaction amount compared with the account balance.

Amount_Zscore_User
Transaction amount anomaly score based on user behavior.


FEATURE ENGINEERING
===================

Additional date-based features are extracted from the Date column.

These features include:

- Year
- Month
- Day
- DayOfWeek
- Hour

The project also uses transaction-related engineered features:

- Txn_to_Balance_Pct
- Amount_Zscore_User


MACHINE LEARNING MODEL
======================

The final model used in this project is:

Random Forest Classifier


MODEL INPUT FEATURES
====================

NUMERICAL FEATURES

- Transaction_Amount
- Account_Balance
- Previous_Fraudulent_Activity
- Daily_Transaction_Count
- Card_Age
- Txn_to_Balance_Pct
- Amount_Zscore_User
- Year
- Month
- Day
- DayOfWeek
- Hour


CATEGORICAL FEATURES

- Transaction_Type
- Device_Type
- Location
- Merchant_Category
- Card_Type


FRAUD THRESHOLD OPTIMIZATION
============================

Fraud detection can be affected by class imbalance.

Using the default classification threshold of 0.50 resulted in low fraud detection recall.

Therefore, multiple thresholds were evaluated.

The selected fraud decision threshold is:

Fraud Decision Threshold = 0.17

This threshold provides a better balance between:

- Precision
- Recall
- F1 Score

The threshold of 0.17 is used by the final fraud prediction system.


FINAL MODEL PERFORMANCE
=======================

The final model achieved approximately:

Accuracy  : 88.64%
Precision : 34.56%
Recall    : 48.42%
F1 Score  : 40.34%
ROC-AUC   : 0.7273

Fraud Decision Threshold : 0.17


CONFUSION MATRIX
================

[[8480, 727],
 [ 409, 384]]

Where:

True Negatives  = 8480
False Positives = 727
False Negatives = 409
True Positives  = 384


STREAMLIT DASHBOARD
===================

The project includes an interactive Streamlit dashboard.

The dashboard contains the following sections:


1. OVERVIEW
===========

The Overview page provides a high-level summary of the fraud dataset.

Key metrics include:

- Total Transactions
- Fraud Transactions
- Fraud Rate
- Fraud Amount
- Fraud Distribution
- Transaction Trends


2. FRAUD ANALYSIS
=================

The Fraud Analysis page allows users to explore fraud patterns using interactive filters.

Available filters include:

- Transaction Type
- Device Type
- Location
- Merchant Category
- Date Range

Users can analyze fraud patterns based on the selected filters.


3. FRAUD PREDICTION
===================

The Fraud Prediction page allows users to enter transaction information and receive a fraud-risk prediction.

The system provides:

- Fraud Probability
- Fraud Prediction
- Fraud Decision Threshold
- Risk Level

The optimized fraud threshold used for prediction is:

0.17


RISK LEVELS
===========

Low Risk

Medium Risk

High Risk

Critical Risk


4. MODEL PERFORMANCE
====================

The Model Performance page displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve


TECHNOLOGIES USED
=================

Python
Pandas
NumPy
Scikit-learn
Streamlit
Plotly
Joblib
Jupyter Notebook


INSTALLATION
============

1. Clone the repository

git clone https://github.com/YOUR_GITHUB_USERNAME/Financial-Fraud-Detection.git


2. Navigate to the project directory

cd Financial-Fraud-Detection


3. Create a virtual environment

python -m venv .venv


4. Activate the virtual environment

For Windows Command Prompt:

.venv\Scripts\activate


For Git Bash:

source .venv/Scripts/activate


5. Install dependencies

pip install -r requirements.txt


RUNNING THE PROJECT
===================

Generate Dataset

python src/generate_fraud_dataset.py


Run Dataset Diagnostics

python src/diagnose_fraud_data.py


This script analyzes:

- Dataset shape
- Missing values
- Duplicate records
- Fraud distribution
- Categorical fraud patterns
- Numerical feature analysis
- Correlation with fraud labels


Analyze Target Dependency

python src/check_target_dependency.py


This script performs:

- Mutual Information analysis
- Target dependency analysis
- Target shuffle testing


Train the Final Model

python src/train_final_model.py


This generates:

models/fraud_model.pkl
models/preprocessor.pkl
models/feature_information.pkl
data/processed/test_data.csv


Optimize the Fraud Threshold

python src/optimize_threshold.py


This script evaluates multiple fraud classification thresholds and selects the best threshold based on model performance.

Selected threshold:

0.17


Launch the Streamlit Dashboard

streamlit run dashboard/app.py


JUPYTER NOTEBOOKS
=================

01_Data_Exploration.ipynb

Performs initial dataset exploration and exploratory data analysis.


02_Data_Preprocessing.ipynb

Performs data cleaning and preprocessing.


03_Model_Training.ipynb

Trains and compares Machine Learning models.


04_Model_Evaluation.ipynb

Evaluates model performance using classification metrics and visualizations.


REPORTS
=======

The project generates the following reports:

reports/feature_importance.csv

Contains information about the importance of model features.


reports/final_model_evaluation.csv

Contains the final model evaluation metrics.


reports/model_comparison.csv

Contains a comparison of different Machine Learning models.


reports/model_predictions.csv

Contains transaction prediction results.


FUTURE IMPROVEMENTS
===================

Possible future improvements include:

- Real-time transaction fraud detection.
- Live transaction stream integration.
- Hyperparameter optimization.
- Gradient Boosting and XGBoost model comparison.
- Advanced anomaly detection.
- SHAP-based model explainability.
- Real-time fraud alerts.
- Database integration.
- REST API for fraud prediction.
- Cloud deployment.
- User authentication.
- Role-based access control.


AUTHOR
======

Anil Kandarapu

B.Tech Graduate
Data Science and Machine Learning Enthusiast


LICENSE
=======

This project is intended for:

- Educational purposes
- Learning purposes
- Internship purposes
- Portfolio purposes


PROJECT SUMMARY
===============

The Financial Fraud Detection System demonstrates an end-to-end Machine Learning workflow for fraud-risk analysis.

The project includes:

- Dataset exploration
- Data diagnostics
- Feature engineering
- Data preprocessing
- Random Forest classification
- Model evaluation
- Threshold optimization
- Fraud-risk prediction
- Interactive Streamlit visualization

The final system uses an optimized fraud decision threshold of 0.17 to provide a better balance between fraud detection recall and precision.
