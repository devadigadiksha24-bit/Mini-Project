# IntelliPredict MLOps

## An Automated AI Platform for Customer Churn Prediction, Model Optimization and Continuous Deployment

IntelliPredict is an end-to-end MLOps project that predicts customer churn using machine learning and automates the complete ML lifecycle using DVC, MLflow, GitHub Actions, Docker, FastAPI and Streamlit.

## Project Objectives

- Predict whether a customer is likely to churn.
- Preprocess and clean customer data automatically.
- Train and compare multiple machine learning models.
- Select the best-performing model using evaluation metrics.
- Track experiments using MLflow.
- Manage the ML pipeline using DVC.
- Automate testing and retraining using GitHub Actions.
- Deploy the prediction API using Docker and FastAPI.
- Provide an interactive Streamlit dashboard.

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MLflow
- DVC
- FastAPI
- Streamlit
- Docker
- Git and GitHub
- GitHub Actions
- Pytest

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract
- Payment Method
- Monthly Charges
- Total Charges
- Churn

## Data Preprocessing

The preprocessing pipeline performs:

1. Removal of the customer ID column.
2. Conversion of TotalCharges into numeric format.
3. Handling of missing TotalCharges values.
4. Conversion of the Churn target into binary values.
5. Removal of duplicate records.
6. Saving the processed dataset.

The processed dataset contains 7,021 records and 20 features.

## Machine Learning

The project compares multiple classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Training Time

The best model is selected automatically based primarily on F1 Score.

## MLflow

MLflow is used to track:

- Model name
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Training time
- Experiment information

This allows different model experiments to be compared and reproduced.

## DVC Pipeline

The DVC pipeline contains two main stages:

```text
Preprocessing
      ↓
Training