import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.ml_models.predictor import ProteinExpressionPredictor
from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator
from typing import Any


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing using real Expression Atlas data"""
    return pd.read_csv("tests/data/test_expression_data.csv")


@pytest.fixture
def mock_generator() -> MockProteinExpressionDataGenerator:
    """Create a mock data generator"""
    return MockProteinExpressionDataGenerator(num_records=10)


def test_predictor_initialization() -> None:
    """Test predictor initialization"""
    predictor = ProteinExpressionPredictor()
    assert predictor.feature_columns == [
        "host_organism",
        "vector_type",
        "induction_condition",
        "media_type",
        "temperature",
        "induction_time",
    ]
    assert predictor.target_columns == ["expression_level", "solubility"]
    assert isinstance(predictor.model, type(RandomForestRegressor()))
    assert predictor.label_encoders == {}


def test_prepare_features(sample_data: pd.DataFrame) -> None:
    """Test feature preparation"""
    predictor = ProteinExpressionPredictor()
    X, y = predictor.prepare_features(sample_data, is_training=True)

    # Check shapes
    assert X.shape == (len(sample_data), 6)  # N samples, 6 features
    if y is not None:  # Add type check for y
        assert y.shape == (len(sample_data), 2)  # N samples, 2 targets

    # Check numeric features remain unchanged
    np.testing.assert_array_almost_equal(
        X[:, -2:],  # temperature and induction_time
        sample_data[["temperature", "induction_time"]].values,
    )

    # Test reusing existing label encoders
    X2, y2 = predictor.prepare_features(sample_data, is_training=True)
    np.testing.assert_array_almost_equal(X, X2)


def test_train_and_predict(sample_data: pd.DataFrame) -> None:
    """Test model training and prediction"""
    predictor = ProteinExpressionPredictor()

    # Train the model
    predictor.train(sample_data)

    # Make predictions
    predictions = predictor.predict(sample_data)

    # Check prediction shape and type
    assert predictions.shape == (len(sample_data), 2)  # N samples, 2 targets
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64

    # Check feature importance
    importance = predictor.get_feature_importance()
    assert len(importance) == 6  # One value per feature
    assert all(isinstance(v, float) for v in importance.values())
    assert all(
        0 <= v <= 1 for v in importance.values()
    )  # Importance scores between 0 and 1


def test_predictor_error_handling(sample_data: pd.DataFrame) -> None:
    """Test error handling"""
    predictor = ProteinExpressionPredictor()

    # Test training with missing target columns
    data_without_targets = sample_data.drop(["expression_level", "solubility"], axis=1)
    with pytest.raises(ValueError, match="Training data must include target columns"):
        predictor.train(data_without_targets)

    # Test prediction with unknown categories
    invalid_data = pd.DataFrame(
        {
            "host_organism": ["Unknown_Organism"],
            "vector_type": ["pET"],
            "induction_condition": ["IPTG"],
            "media_type": ["LB"],
            "temperature": [37.0],
            "induction_time": [4.0],
        }
    )

    # Train first with valid data
    predictor.train(sample_data)

    # Then test with invalid data
    with pytest.raises(ValueError, match="Unknown category in host_organism"):
        predictor.predict(invalid_data)


def test_end_to_end_with_real_data(sample_data: pd.DataFrame) -> None:
    """Test the complete workflow using real Expression Atlas data"""
    # Split data into train and test
    train_data = sample_data.iloc[:6]  # First 6 samples for training
    test_data = sample_data.iloc[6:]  # Last 3 samples for testing

    # Train predictor
    predictor = ProteinExpressionPredictor()
    predictor.train(train_data)

    # Make predictions
    predictions = predictor.predict(test_data)

    # Check predictions
    assert predictions.shape == (len(test_data), 2)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 100)

    # Check feature importance
    importance = predictor.get_feature_importance()
    assert len(importance) == 6
    assert abs(sum(importance.values()) - 1.0) < 1e-6  # Should sum to approximately 1

    # Test prediction with new data using known categories
    single_prediction = predictor.predict(
        pd.DataFrame(
            {
                "host_organism": [test_data["host_organism"].iloc[0]],
                "vector_type": [test_data["vector_type"].iloc[0]],
                "induction_condition": [test_data["induction_condition"].iloc[0]],
                "media_type": [test_data["media_type"].iloc[0]],
                "temperature": [37.0],
                "induction_time": [24.0],
            }
        )
    )
    assert single_prediction.shape == (1, 2)
    assert np.all(single_prediction >= 0)
    assert np.all(single_prediction <= 100)


def test_prepare_features_edge_cases() -> None:
    """Test edge cases in feature preparation"""
    predictor = ProteinExpressionPredictor()

    # Test with empty DataFrame
    empty_df = pd.DataFrame(
        columns=[
            "host_organism",
            "vector_type",
            "induction_condition",
            "media_type",
            "temperature",
            "induction_time",
        ]
    )
    X, y = predictor.prepare_features(empty_df, is_training=True)
    assert X.shape == (0, 6)
    assert y is None

    # Test with single row
    single_row = pd.DataFrame(
        {
            "host_organism": ["E. coli"],
            "vector_type": ["pET"],
            "induction_condition": ["IPTG"],
            "media_type": ["LB"],
            "temperature": [37.0],
            "induction_time": [4.0],
        }
    )
    X, y = predictor.prepare_features(single_row, is_training=True)
    assert X.shape == (1, 6)
    assert y is None
