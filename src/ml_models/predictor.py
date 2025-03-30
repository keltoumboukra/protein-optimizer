import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Tuple

class ProteinExpressionPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = [
            "host_organism", "vector_type", "induction_condition", 
            "media_type", "temperature", "induction_time"
        ]
        self.target_columns = ["expression_level", "solubility"]
        
    def prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for the model."""
        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Create and fit label encoders for categorical features
        categorical_features = ["host_organism", "vector_type", "induction_condition", "media_type"]
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                if is_training:
                    df_copy[feature] = self.label_encoders[feature].fit_transform(df_copy[feature])
                else:
                    # For prediction, we need to handle unseen categories
                    try:
                        df_copy[feature] = self.label_encoders[feature].transform(df_copy[feature])
                    except ValueError as e:
                        raise ValueError(f"Unknown category in {feature}. Please use one of: {self.label_encoders[feature].classes_}")
            else:
                # Reuse existing label encoder
                if is_training:
                    df_copy[feature] = self.label_encoders[feature].fit_transform(df_copy[feature])
                else:
                    try:
                        df_copy[feature] = self.label_encoders[feature].transform(df_copy[feature])
                    except ValueError as e:
                        raise ValueError(f"Unknown category in {feature}. Please use one of: {self.label_encoders[feature].classes_}")
        
        # Extract features
        X = df_copy[self.feature_columns].values
        
        # For training, we need the target variables
        if all(col in df_copy.columns for col in self.target_columns):
            y = df_copy[self.target_columns].values
        else:
            y = None
            
        return X, y
    
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data."""
        X, y = self.prepare_features(df, is_training=True)
        if y is not None:
            self.model.fit(X, y)
        else:
            raise ValueError(f"Training data must include target columns: {self.target_columns}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expression level and solubility for new experiments."""
        X, _ = self.prepare_features(df, is_training=False)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))

if __name__ == "__main__":
    # Example usage
    from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator
    
    # Generate training data
    generator = MockProteinExpressionDataGenerator(num_records=1000)
    train_data = generator.generate()
    
    # Train model
    predictor = ProteinExpressionPredictor()
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
    for i, (actual, predicted) in enumerate(zip(test_data[self.target_columns].values, predictions)):
        print(f"Experiment {i+1}:")
        print(f"  Actual: Expression={actual[0]:.2f}%, Solubility={actual[1]:.2f}%")
        print(f"  Predicted: Expression={predicted[0]:.2f}%, Solubility={predicted[1]:.2f}%") 