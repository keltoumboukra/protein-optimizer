import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple

class BugPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for the model."""
        # Create label encoders for categorical features
        categorical_features = ["instrument", "problem_type", "severity", "status"]
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        # Extract features
        feature_columns = ["instrument", "problem_type", "severity", "status"]
        X = df[feature_columns].values
        y = df["resolution_time_hours"].values
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data."""
        X, y = self.prepare_features(df)
        self.model.fit(X, y)
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict resolution time for new bugs."""
        X, _ = self.prepare_features(df)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        feature_columns = ["instrument", "problem_type", "severity", "status"]
        importance = self.model.feature_importances_
        return dict(zip(feature_columns, importance))

if __name__ == "__main__":
    # Example usage
    from src.data_pipeline.mock_data import MockBugDataGenerator
    
    # Generate training data
    generator = MockBugDataGenerator(num_records=1000)
    train_data = generator.generate()
    
    # Train model
    predictor = BugPredictor()
    predictor.train(train_data)
    
    # Generate test data
    test_data = generator.generate(num_records=10)
    
    # Make predictions
    predictions = predictor.predict(test_data)
    
    # Print results
    print("\nFeature Importance:")
    for feature, importance in predictor.get_feature_importance().items():
        print(f"{feature}: {importance:.4f}")
    
    print("\nSample Predictions:")
    for i, (actual, predicted) in enumerate(zip(test_data["resolution_time_hours"], predictions)):
        print(f"Bug {i+1}: Actual={actual:.2f}h, Predicted={predicted:.2f}h") 