# Financial Fraud Detection System

## Project Overview

The Financial Fraud Detection System is a machine learning and data analytics project designed to identify potentially fraudulent financial transactions.

The system analyzes transaction characteristics and uses a machine learning classification model to predict whether a transaction is legitimate or fraudulent.

An interactive Streamlit dashboard is provided for data exploration, fraud analysis, transaction prediction, and model performance evaluation.

---

# Objectives

The main objectives of this project are:

- Identify fraudulent financial transactions.
- Analyze patterns associated with fraudulent activity.
- Build a machine learning classification model.
- Evaluate the model using standard classification metrics.
- Provide an interactive dashboard for fraud monitoring.
- Allow users to enter transaction details and obtain fraud predictions.
- Generate fraud transaction reports.

---

# Technology Stack

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- Random Forest Classifier

## Visualization

- Plotly

## Dashboard

- Streamlit

## Model Persistence

- Joblib

---

# Project Structure

```text
Financial_Fraud_Detection/
│
├── data/
│   ├── raw/
│   │   └── synthetic_fraud_dataset.csv
│   │
│   └── processed/
│       └── test_data.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   └── feature_information.pkl
│
├── dashboard/
│   └── app.py
│
├── src/
│   └── train_final_model.py
│
├── notebooks/
│
├── reports/
│
├── requirements.txt
│
└── README.md