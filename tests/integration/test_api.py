import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root() -> None:
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Protein Expression Optimization API"
    }


def test_valid_categories() -> None:
    """Test valid categories endpoint"""
    response = client.get("/valid-categories")
    assert response.status_code == 200
    data = response.json()

    # Check all required categories are present
    assert all(
        key in data
        for key in ["host_organism", "vector_type", "induction_condition", "media_type"]
    )

    # Check specific values
    assert "E. coli" in data["host_organism"]
    assert "pET" in data["vector_type"]
    assert "IPTG" in data["induction_condition"]
    assert "LB" in data["media_type"]


def test_generate_sample() -> None:
    """Test sample generation endpoint"""
    response = client.get("/generate-sample")
    assert response.status_code == 200
    data = response.json()

    # Check all required fields are present
    required_fields = [
        "host_organism",
        "vector_type",
        "induction_condition",
        "media_type",
        "temperature",
        "induction_time",
        "expression_level",
        "solubility",
    ]
    assert all(field in data for field in required_fields)

    # Check value ranges
    assert 20 <= data["temperature"] <= 37
    assert 2 <= data["induction_time"] <= 24
    assert 0 <= data["expression_level"] <= 100
    assert 0 <= data["solubility"] <= 100


def test_predict_endpoint() -> None:
    """Test prediction endpoint"""
    # Test valid input
    valid_input = {
        "host_organism": "E. coli",
        "vector_type": "pET",
        "induction_condition": "IPTG",
        "media_type": "LB",
        "temperature": 37.0,
        "induction_time": 4.0,
        "description": "Test prediction",
    }
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_expression_level" in data
    assert "predicted_solubility" in data
    assert "feature_importance" in data

    # Test invalid input
    invalid_input = {
        "host_organism": "Invalid_Organism",  # Invalid value
        "vector_type": "pET",
        "induction_condition": "IPTG",
        "media_type": "LB",
        "temperature": 37.0,
        "induction_time": 4.0,
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 400  # Bad request
