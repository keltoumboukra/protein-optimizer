"""Tests for the prediction history functionality."""

import os
import json
import pytest
import pandas as pd
from datetime import datetime
import time
from dashboard.prediction_history import PredictionHistory


@pytest.fixture
def temp_storage_path(tmp_path):
    """Create a temporary storage path for testing."""
    return str(tmp_path / "test_predictions")


@pytest.fixture
def prediction_history(temp_storage_path):
    """Create a PredictionHistory instance with temporary storage."""
    return PredictionHistory(storage_path=temp_storage_path)


@pytest.fixture
def sample_prediction_data():
    """Create sample prediction data for testing."""
    return {
        "host_organism": "E. coli",
        "vector_type": "pET",
        "induction_condition": "IPTG",
        "media_type": "LB",
        "temperature": 37.0,
        "induction_time": 4.0,
        "description": "Test prediction",
        "predicted_expression_level": 85.5,
        "predicted_solubility": 90.0,
        "feature_importance": {"temperature": 0.3, "induction_time": 0.7},
    }


def test_save_prediction(prediction_history, sample_prediction_data, temp_storage_path):
    """Test saving a prediction to history."""
    prediction_history.save_prediction(sample_prediction_data)
    
    # Check that a file was created
    files = os.listdir(temp_storage_path)
    assert len(files) == 1
    
    # Check file contents
    with open(os.path.join(temp_storage_path, files[0]), "r") as f:
        saved_data = json.load(f)
        assert "timestamp" in saved_data
        assert saved_data["host_organism"] == sample_prediction_data["host_organism"]
        assert saved_data["predicted_expression_level"] == sample_prediction_data["predicted_expression_level"]


def test_load_history(prediction_history, sample_prediction_data, temp_storage_path):
    """Test loading prediction history."""
    # Save multiple predictions with a small delay between each
    saved_files = []
    for i in range(3):
        prediction_data = sample_prediction_data.copy()
        prediction_data["description"] = f"Test prediction {i+1}"
        prediction_history.save_prediction(prediction_data)
        time.sleep(0.1)  # Add a small delay to ensure unique timestamps
        
        # Print debug info
        print(f"\nAfter saving prediction {i+1}:")
        files = os.listdir(temp_storage_path)
        print(f"Files in directory: {files}")
        saved_files.extend(files)
    
    # Print final state
    print(f"\nAll saved files: {saved_files}")
    print(f"Files in directory: {os.listdir(temp_storage_path)}")
    
    # Load history
    history_df = prediction_history.load_history()
    
    # Print DataFrame info
    print(f"\nDataFrame info:")
    print(f"Shape: {history_df.shape}")
    print(f"Columns: {history_df.columns.tolist()}")
    print(f"Records:\n{history_df}")
    
    # Check DataFrame
    assert isinstance(history_df, pd.DataFrame)
    assert len(history_df) == 3, f"Expected 3 records, but got {len(history_df)}"
    assert "timestamp" in history_df.columns
    assert "host_organism" in history_df.columns
    assert "predicted_expression_level" in history_df.columns


def test_load_empty_history(prediction_history):
    """Test loading history when no predictions exist."""
    history_df = prediction_history.load_history()
    assert isinstance(history_df, pd.DataFrame)
    assert history_df.empty


def test_export_history_csv(prediction_history, sample_prediction_data):
    """Test exporting history to CSV format."""
    # Save a prediction
    prediction_history.save_prediction(sample_prediction_data)
    
    # Export to CSV
    filepath = prediction_history.export_history(format="csv")
    
    # Check file
    assert filepath is not None
    assert os.path.exists(filepath)
    assert filepath.endswith(".csv")
    
    # Check contents
    df = pd.read_csv(filepath)
    assert len(df) == 1
    assert "host_organism" in df.columns


def test_export_history_json(prediction_history, sample_prediction_data):
    """Test exporting history to JSON format."""
    # Save a prediction
    prediction_history.save_prediction(sample_prediction_data)
    
    # Export to JSON
    filepath = prediction_history.export_history(format="json")
    
    # Check file
    assert filepath is not None
    assert os.path.exists(filepath)
    assert filepath.endswith(".json")
    
    # Check contents
    with open(filepath, "r") as f:
        data = json.load(f)
        assert len(data) == 1
        assert "host_organism" in data[0]


def test_export_empty_history(prediction_history):
    """Test exporting history when no predictions exist."""
    filepath = prediction_history.export_history(format="csv")
    assert filepath is None
