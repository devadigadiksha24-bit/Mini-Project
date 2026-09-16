import streamlit as st
import pandas as pd
import requests
import json
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IntelliPredict MLOps",
    page_icon=None,
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "churn_processed.csv"
)

MODEL_COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "metrics.json"
)

API_URL = "http://127.0.0.1:8000"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset():

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    return pd.DataFrame()


@st.cache_data
def load_model_comparison():

    if os.path.exists(MODEL_COMPARISON_PATH):
        return pd.read_csv(MODEL_COMPARISON_PATH)

    return pd.DataFrame()


def load_metrics():

    if os.path.exists(METRICS_PATH):

        with open(METRICS_PATH, "r") as file:
            return json.load(file)

    return {}


df = load_dataset()
comparison = load_model_comparison()
metrics = load_metrics()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("IntelliPredict")

st.sidebar.caption(
    "Customer Churn Prediction and MLOps Platform"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Predict Churn",
        "Model Comparison",
        "Analytics",
        "MLOps Monitor",
        "About"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("IntelliPredict MLOps")

    st.write(
        "A machine learning platform for analyzing customer "
        "churn and generating churn predictions."
    )

    st.markdown("---")

    total_customers = len(df)

    if "Churn" in df.columns and total_customers > 0:

        churned = df["Churn"].sum()

        churn_rate = (
            churned / total_customers
        ) * 100

    else:

        churn_rate = 0

    best_model = "Not available"
    best_f1 = 0

    if not comparison.empty:

        if "F1" in comparison.columns:

            best_row = comparison.loc[
                comparison["F1"].idxmax()
            ]

            best_model = str(
                best_row.get(
                    "Model",
                    "Not available"
                )
            )

            best_f1 = float(
                best_row["F1"]
            )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col3:

        st.metric(
            "Selected Model",
            best_model
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{best_f1:.4f}"
        )

    st.markdown("---")

    if "Churn" in df.columns:

        st.subheader("Customer Churn Overview")

        churn_counts = df["Churn"].value_counts()

        chart_data = pd.DataFrame({
            "Customer Status": [
                "No Churn" if value == 0 else "Churn"
                for value in churn_counts.index
            ],
            "Customers": churn_counts.values
        })

        st.bar_chart(
            chart_data.set_index(
                "Customer Status"
            )
        )

    if not comparison.empty:

        st.subheader("Model Performance")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# PREDICT CHURN
# =========================================================

elif page == "Predict Churn":

    st.title("Customer Churn Prediction")

    st.write(
        "Enter the customer's details below to estimate "
        "their likelihood of leaving the service."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["No", "Yes"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12
            )

        with col2:

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service"
                ]
            )

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            security = st.selectbox(
                "Online Security",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            backup = st.selectbox(
                "Online Backup",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            device = st.selectbox(
                "Device Protection",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

        with col3:

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "No",
                    "Yes",
                    "No internet service"
                ]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        st.subheader("Billing Information")

        col1, col2 = st.columns(2)

        with col1:

            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

        with col2:

            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Generate Prediction"
        )

    # =====================================================
    # API REQUEST
    # =====================================================

    if submitted:

        customer = {

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=customer,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Prediction generated successfully."
                )

                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Churn Prediction",
                        result["churn"]
                    )

                with col2:

                    st.metric(
                        "Churn Probability",
                        f"{result['churn_percentage']:.2f}%"
                    )

                with col3:

                    st.metric(
                        "Risk Level",
                        result["risk_level"]
                    )

                st.subheader("Prediction Details")

                st.write(
                    f"The model estimates a "
                    f"{result['churn_percentage']:.2f}% "
                    f"probability of churn for this customer."
                )

            else:

                st.error(
                    f"The prediction service returned "
                    f"status code {response.status_code}."
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "The prediction service is not available. "
                "Please start FastAPI before generating "
                "a prediction."
            )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.title("Model Comparison")

    st.write(
        "Performance comparison of the machine learning "
        "models trained for customer churn prediction."
    )

    st.markdown("---")

    if comparison.empty:

        st.warning(
            "Model comparison data is not available."
        )

    else:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        numeric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]

        available_columns = [
            column
            for column in numeric_columns
            if column in comparison.columns
        ]

        if available_columns:

            st.subheader("Performance Metrics")

            chart_data = comparison.set_index(
                "Model"
            )[available_columns]

            st.bar_chart(chart_data)


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.title("Customer Analytics")

    st.write(
        "Basic analysis of the processed customer dataset."
    )

    st.markdown("---")

    if df.empty:

        st.warning(
            "The processed dataset could not be found."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Dataset Rows",
                f"{df.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Dataset Columns",
                df.shape[1]
            )

        st.subheader("Churn Distribution")

        if "Churn" in df.columns:

            churn_data = (
                df["Churn"]
                .value_counts()
                .rename(
                    index={
                        0: "No Churn",
                        1: "Churn"
                    }
                )
            )

            st.bar_chart(churn_data)

        if "Contract" in df.columns:

            st.subheader(
                "Customers by Contract Type"
            )

            contract_data = (
                df["Contract"]
                .value_counts()
            )

            st.bar_chart(contract_data)


# =========================================================
# MLOPS MONITOR
# =========================================================

elif page == "MLOps Monitor":

    st.title("MLOps Monitor")

    st.write(
        "Current project components and generated artifacts."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "DVC",
            "Configured"
        )

    with col2:

        st.metric(
            "MLflow",
            "Configured"
        )

    with col3:

        st.metric(
            "FastAPI",
            "Available"
        )

    st.subheader("Project Artifacts")

    artifacts = {

        "Processed Dataset": os.path.exists(
            DATA_PATH
        ),

        "Best Model": os.path.exists(
            os.path.join(
                BASE_DIR,
                "models",
                "best_model.pkl"
            )
        ),

        "Metrics File": os.path.exists(
            METRICS_PATH
        ),

        "Model Comparison": os.path.exists(
            MODEL_COMPARISON_PATH
        ),

        "DVC Pipeline": os.path.exists(
            os.path.join(
                BASE_DIR,
                "dvc.yaml"
            )
        ),

        "MLflow Runs": os.path.exists(
            os.path.join(
                BASE_DIR,
                "mlruns"
            )
        )
    }

    for artifact, status in artifacts.items():

        if status:

            st.success(
                f"{artifact}: Available"
            )

        else:

            st.warning(
                f"{artifact}: Not found"
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title("About IntelliPredict")

    st.write(
        """
        IntelliPredict is a machine learning project developed
        for customer churn prediction using an MLOps workflow.

        The system processes customer data, trains multiple
        machine learning models, compares their performance,
        selects a model, and provides predictions through a
        FastAPI service.

        A Streamlit dashboard is used to present predictions,
        model performance, customer analytics, and project
        information.
        """
    )

    st.subheader("Project Workflow")

    st.write(
        """
        Data Collection → Data Preprocessing → Model Training
        → MLflow Tracking → Model Comparison → Best Model
        → FastAPI → Streamlit Dashboard
        """
    )

    st.subheader("Technologies Used")

    st.write(
        """
        Python, Pandas, Scikit-learn, XGBoost, MLflow,
        DVC, FastAPI, Streamlit, Git and GitHub.
        """
    )