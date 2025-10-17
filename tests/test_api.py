from fastapi.testclient import TestClient
from app import app  
import os
import pytest

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Fixture to ensure the model is trained before tests run."""
    if not os.path.exists("models/model_pipeline.joblib"):
        print("Model not found. Running training script...")
        os.system('python -c "from sklearn.datasets import fetch_california_housing; import pandas as pd; housing = fetch_california_housing(); df = pd.DataFrame(data=housing.data, columns=housing.feature_names); df[\'MedHouseVal\'] = housing.target; df.to_csv(\'data/california_housing.csv\', index=False)"')
        os.system("python train.py")
    yield

def test_health_check():
    """Test the root/health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

def test_prediction():
    """Test a valid prediction."""
    payload = {
        "MedInc": 3.8716,
        "HouseAge": 25.0,
        "AveRooms": 6.284,
        "AveBedrms": 1.08,
        "Population": 1200.0,
        "AveOccup": 2.54,
        "Latitude": 34.23,
        "Longitude": -118.49
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0

def test_bad_input():
    """Test a request with missing data."""
    payload = {
        "MedInc": 3.8716,
        "HouseAge": 25.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422