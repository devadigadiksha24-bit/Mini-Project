from fastapi.testclient import TestClient
from api.app import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "IntelliPredict MLOps"
    assert data["status"] == "running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70,
        "TotalCharges": 840
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn" in data
    assert "churn_probability" in data
    assert "churn_percentage" in data
    assert "risk_level" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1
    assert 0 <= data["churn_percentage"] <= 100
    assert data["risk_level"] in [
        "Low",
        "Medium",
        "High"
    ]