import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.ml_models.predictor import ProteinExpressionPredictor
from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        "host_organism": ["E. coli", "S. cerevisiae"],
        "vector_type": ["pET", "pGEX"],
        "induction_condition": ["IPTG", "Galactose"],
        "media_type": ["LB", "YPD"],
        "temperature": [37.0, 30.0],
        "induction_time": [4.0, 12.0],
        "expression_level": [75.0, 60.0],
        "solubility": [80.0, 65.0]
    })

@pytest.fixture
def mock_generator():
    """Create a mock data generator"""
    return MockProteinExpressionDataGenerator(num_records=10)

def test_predictor_initialization():
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

def test_prepare_features(sample_data):
    """Test feature preparation"""
    predictor = ProteinExpressionPredictor()
    X, y = predictor.prepare_features(sample_data, is_training=True)
    
    # Check shapes
    assert X.shape == (2, 6)  # 2 samples, 6 features
    assert y.shape == (2, 2)  # 2 samples, 2 targets
    
    # Check numeric features remain unchanged
    np.testing.assert_array_almost_equal(
        X[:, -2:],  # temperature and induction_time
        sample_data[["temperature", "induction_time"]].values
    )

    # Test reusing existing label encoders
    X2, y2 = predictor.prepare_features(sample_data, is_training=True)
    np.testing.assert_array_almost_equal(X, X2)

def test_train_and_predict(sample_data):
    """Test model training and prediction"""
    predictor = ProteinExpressionPredictor()
    
    # Train the model
    predictor.train(sample_data)
    
    # Make predictions
    predictions = predictor.predict(sample_data)
    
    # Check prediction shape and type
    assert predictions.shape == (2, 2)  # 2 samples, 2 targets
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64
    
    # Check feature importance
    importance = predictor.get_feature_importance()
    assert len(importance) == 6  # One value per feature
    assert all(isinstance(v, float) for v in importance.values())
    assert all(0 <= v <= 1 for v in importance.values())  # Importance scores between 0 and 1

def test_predictor_error_handling(sample_data):
    """Test error handling"""
    predictor = ProteinExpressionPredictor()
    
    # Test training with missing target columns
    data_without_targets = sample_data.drop(["expression_level", "solubility"], axis=1)
    with pytest.raises(ValueError, match="Training data must include target columns"):
        predictor.train(data_without_targets)
    
    # Test prediction with unknown categories
    invalid_data = pd.DataFrame({
        "host_organism": ["Unknown_Organism"],
        "vector_type": ["pET"],
        "induction_condition": ["IPTG"],
        "media_type": ["LB"],
        "temperature": [37.0],
        "induction_time": [4.0]
    })
    
    # Train first with valid data
    predictor.train(sample_data)
    
    # Then test with invalid data
    with pytest.raises(ValueError, match="Unknown category in host_organism"):
        predictor.predict(invalid_data)

def test_end_to_end_with_mock_data(mock_generator):
    """Test the complete workflow using mock data"""
    # Generate training data and save the first generated sample for testing
    train_data = mock_generator.generate(num_records=100)
    
    # Create test data using only categories from training data
    test_data = pd.DataFrame({
        "host_organism": train_data["host_organism"].iloc[:5],
        "vector_type": train_data["vector_type"].iloc[:5],
        "induction_condition": train_data["induction_condition"].iloc[:5],
        "media_type": train_data["media_type"].iloc[:5],
        "temperature": train_data["temperature"].iloc[:5],
        "induction_time": train_data["induction_time"].iloc[:5],
        "expression_level": train_data["expression_level"].iloc[:5],
        "solubility": train_data["solubility"].iloc[:5]
    })
    
    # Train predictor
    predictor = ProteinExpressionPredictor()
    predictor.train(train_data)
    
    # Make predictions
    predictions = predictor.predict(test_data)
    
    # Check predictions
    assert predictions.shape == (5, 2)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 100)
    
    # Check feature importance
    importance = predictor.get_feature_importance()
    assert len(importance) == 6
    assert abs(sum(importance.values()) - 1.0) < 1e-6  # Should sum to approximately 1

    # Test prediction with new data using known categories
    single_prediction = predictor.predict(pd.DataFrame({
        "host_organism": [train_data["host_organism"].iloc[0]],
        "vector_type": [train_data["vector_type"].iloc[0]],
        "induction_condition": [train_data["induction_condition"].iloc[0]],
        "media_type": [train_data["media_type"].iloc[0]],
        "temperature": [37.0],
        "induction_time": [4.0]
    }))

def test_prepare_features_edge_cases():
    """Test edge cases in feature preparation"""
    predictor = ProteinExpressionPredictor()
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame(columns=[
        "host_organism", "vector_type", "induction_condition",
        "media_type", "temperature", "induction_time"
    ])
    X, y = predictor.prepare_features(empty_df, is_training=True)
    assert X.shape == (0, 6)
    assert y is None
    
    # Test with single row
    single_row = pd.DataFrame({
        "host_organism": ["E. coli"],
        "vector_type": ["pET"],
        "induction_condition": ["IPTG"],
        "media_type": ["LB"],
        "temperature": [37.0],
        "induction_time": [4.0]
    })
    X, y = predictor.prepare_features(single_row, is_training=True)
    assert X.shape == (1, 6)
    assert y is None 