from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(
    title="IntelliPredict Churn Prediction API",
    description="Customer Churn Prediction Service",
    version="1.0.0"
)

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "best_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Please run src/train.py first."
    )

model = joblib.load(MODEL_PATH)


# ==========================================
# INPUT DATA MODEL
# ==========================================

class CustomerData(BaseModel):

    gender: str = "Female"

    SeniorCitizen: int = Field(
        default=0,
        ge=0,
        le=1
    )

    Partner: str = "No"
    Dependents: str = "No"

    tenure: int = Field(
        default=12,
        ge=0
    )

    PhoneService: str = "Yes"
    MultipleLines: str = "No"

    InternetService: str = "DSL"

    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"

    StreamingTV: str = "No"
    StreamingMovies: str = "No"

    Contract: str = "Month-to-month"

    PaperlessBilling: str = "Yes"

    PaymentMethod: str = "Electronic check"

    MonthlyCharges: float = Field(
        default=70.0,
        ge=0
    )

    TotalCharges: float = Field(
        default=840.0,
        ge=0
    )


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "project": "IntelliPredict MLOps",
        "service": "Customer Churn Prediction API",
        "status": "running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        if hasattr(customer, "model_dump"):
            customer_dict = customer.model_dump()
        else:
            customer_dict = customer.dict()

        input_data = pd.DataFrame([customer_dict])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        if probability >= 0.70:
            risk_level = "High"
        elif probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "prediction": int(prediction),
            "churn": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 4),
            "churn_percentage": round(float(probability) * 100, 2),
            "risk_level": risk_level
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )