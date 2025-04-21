"""
Protein expression optimization model.

This module provides a machine learning model for predicting protein expression levels
and solubility based on experimental conditions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ProteinOptimizer:
    """Machine learning model for protein expression optimization."""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the protein optimizer model.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        self.expression_model = None
        self.solubility_model = None
        self.feature_processor = None
        self.feature_names = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def _create_feature_processor(self, data: pd.DataFrame) -> ColumnTransformer:
        """
        Create a feature processor pipeline for numerical and categorical features.
        
        Args:
            data: Training data with features
            
        Returns:
            ColumnTransformer for feature processing
        """
        # Identify numerical and categorical columns
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def train(self, data: pd.DataFrame, target_columns: List[str]) -> None:
        """
        Train the protein expression and solubility prediction models.
        
        Args:
            data: DataFrame containing features and target variables
            target_columns: List of target column names ['expression_level', 'solubility']
        """
        try:
            # Split features and targets
            X = data.drop(columns=target_columns)
            y_expression = data['expression_level']
            y_solubility = data['solubility']
            
            # Create feature processor
            self.feature_processor = self._create_feature_processor(X)
            self.feature_names = X.columns.tolist()
            
            # Split data into train and validation sets
            X_train, X_val, y_train_exp, y_val_exp = train_test_split(
                X, y_expression, test_size=0.2, random_state=42
            )
            _, _, y_train_sol, y_val_sol = train_test_split(
                X, y_solubility, test_size=0.2, random_state=42
            )
            
            # Create and train expression level model
            self.expression_model = Pipeline([
                ('preprocessor', self.feature_processor),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
            self.expression_model.fit(X_train, y_train_exp)
            
            # Create and train solubility model
            self.solubility_model = Pipeline([
                ('preprocessor', self.feature_processor),
                ('regressor', RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
            self.solubility_model.fit(X_train, y_train_sol)
            
            # Evaluate models
            exp_score = self.expression_model.score(X_val, y_val_exp)
            sol_score = self.solubility_model.score(X_val, y_val_sol)
            
            logger.info(f"Expression model R² score: {exp_score:.3f}")
            logger.info(f"Solubility model R² score: {sol_score:.3f}")
            
            # Save models
            self.save_models()
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Make predictions for protein expression level and solubility.
        
        Args:
            features: DataFrame containing experimental conditions
            
        Returns:
            Tuple of (predicted_expression_level, predicted_solubility)
        """
        try:
            if self.expression_model is None or self.solubility_model is None:
                raise ValueError("Models not trained. Call train() first.")
            
            # Make predictions
            expression_pred = self.expression_model.predict(features)[0]
            solubility_pred = self.solubility_model.predict(features)[0]
            
            return expression_pred, solubility_pred
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the models.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        try:
            if self.expression_model is None or self.solubility_model is None:
                raise ValueError("Models not trained. Call train() first.")
            
            # Get feature names after preprocessing
            feature_names = (
                self.feature_processor.named_transformers_['num'].get_feature_names_out().tolist() +
                self.feature_processor.named_transformers_['cat'].get_feature_names_out().tolist()
            )
            
            # Get feature importance from both models
            exp_importance = self.expression_model.named_steps['regressor'].feature_importances_
            sol_importance = self.solubility_model.named_steps['regressor'].feature_importances_
            
            # Combine importance scores
            importance_dict = {}
            for name, exp_imp, sol_imp in zip(feature_names, exp_importance, sol_importance):
                importance_dict[name] = (exp_imp + sol_imp) / 2
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        try:
            if self.expression_model is None or self.solubility_model is None:
                raise ValueError("No trained models to save")
            
            # Save models
            joblib.dump(self.expression_model, os.path.join(self.model_dir, 'expression_model.joblib'))
            joblib.dump(self.solubility_model, os.path.join(self.model_dir, 'solubility_model.joblib'))
            
            # Save feature processor and names
            joblib.dump(self.feature_processor, os.path.join(self.model_dir, 'feature_processor.joblib'))
            joblib.dump(self.feature_names, os.path.join(self.model_dir, 'feature_names.joblib'))
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        try:
            # Load models
            self.expression_model = joblib.load(os.path.join(self.model_dir, 'expression_model.joblib'))
            self.solubility_model = joblib.load(os.path.join(self.model_dir, 'solubility_model.joblib'))
            
            # Load feature processor and names
            self.feature_processor = joblib.load(os.path.join(self.model_dir, 'feature_processor.joblib'))
            self.feature_names = joblib.load(os.path.join(self.model_dir, 'feature_names.joblib'))
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def optimize(self, 
                target_protein: str,
                optimization_goal: str,
                constraints: Optional[Dict] = None) -> Dict:
        """Generate optimization recommendations for a target protein.
        
        Args:
            target_protein: Name or ID of the protein to optimize
            optimization_goal: Goal of optimization ('expression', 'solubility', or 'both')
            constraints: Optional dictionary of constraints (e.g., {'temperature': (20, 30)})
            
        Returns:
            Dictionary containing recommendations, confidence scores, and explanation
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before optimization")
            
        try:
            # Generate candidate conditions
            candidates = self._generate_candidates(constraints)
            
            # Make predictions
            X = pd.get_dummies(candidates)
            expression_pred = self.expression_model.predict(X)
            solubility_pred = self.solubility_model.predict(X)
            
            # Score candidates based on optimization goal
            if optimization_goal == 'expression':
                scores = expression_pred
            elif optimization_goal == 'solubility':
                scores = solubility_pred
            else:  # 'both'
                scores = (expression_pred + solubility_pred) / 2
                
            # Get top recommendations
            top_indices = np.argsort(scores)[-3:][::-1]
            recommendations = []
            confidence_scores = []
            
            for idx in top_indices:
                rec = {
                    'conditions': candidates.iloc[idx].to_dict(),
                    'predicted_expression': float(expression_pred[idx]),
                    'predicted_solubility': float(solubility_pred[idx])
                }
                recommendations.append(rec)
                confidence_scores.append(float(scores[idx]))
                
            # Generate explanation
            explanation = self._generate_explanation(
                target_protein,
                optimization_goal,
                recommendations[0]
            )
            
            return {
                'recommendations': recommendations,
                'confidence_scores': confidence_scores,
                'explanation': explanation
            }
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise
            
    def _generate_candidates(self, constraints: Optional[Dict] = None) -> pd.DataFrame:
        """Generate candidate experimental conditions.
        
        Args:
            constraints: Optional dictionary of constraints
            
        Returns:
            DataFrame of candidate conditions
        """
        # Default parameter ranges
        param_ranges = {
            'host_organism': ['E. coli', 'S. cerevisiae', 'P. pastoris'],
            'vector_type': ['pET', 'pGEX', 'pMAL'],
            'induction_condition': ['IPTG', 'Arabinose', 'Methanol'],
            'media_type': ['LB', 'TB', 'M9'],
            'temperature': np.arange(20, 42, 2),
            'induction_time': np.arange(2, 24, 2)
        }
        
        # Apply constraints
        if constraints:
            for param, (min_val, max_val) in constraints.items():
                if param in param_ranges:
                    if isinstance(param_ranges[param], list):
                        param_ranges[param] = [x for x in param_ranges[param] 
                                             if min_val <= x <= max_val]
                    else:
                        param_ranges[param] = param_ranges[param][
                            (param_ranges[param] >= min_val) & 
                            (param_ranges[param] <= max_val)
                        ]
                        
        # Generate combinations
        candidates = []
        for host in param_ranges['host_organism']:
            for vector in param_ranges['vector_type']:
                for induction in param_ranges['induction_condition']:
                    for media in param_ranges['media_type']:
                        for temp in param_ranges['temperature']:
                            for time in param_ranges['induction_time']:
                                candidates.append({
                                    'host_organism': host,
                                    'vector_type': vector,
                                    'induction_condition': induction,
                                    'media_type': media,
                                    'temperature': temp,
                                    'induction_time': time
                                })
                                
        return pd.DataFrame(candidates)
        
    def _generate_explanation(self, 
                            target_protein: str,
                            optimization_goal: str,
                            best_conditions: Dict) -> str:
        """Generate a human-readable explanation of the recommendations.
        
        Args:
            target_protein: Name or ID of the target protein
            optimization_goal: Goal of optimization
            best_conditions: Dictionary of the best conditions found
            
        Returns:
            String explaining the recommendations
        """
        conditions = best_conditions['conditions']
        expr_level = best_conditions['predicted_expression']
        solubility = best_conditions['predicted_solubility']
        
        explanation = (
            f"Based on our analysis, for {target_protein}, we recommend:\n"
            f"- Host organism: {conditions['host_organism']}\n"
            f"- Expression vector: {conditions['vector_type']}\n"
            f"- Induction: {conditions['induction_condition']} for {conditions['induction_time']} hours\n"
            f"- Growth conditions: {conditions['media_type']} media at {conditions['temperature']}°C\n\n"
        )
        
        if optimization_goal == 'expression':
            explanation += f"This should yield an expression level of {expr_level:.1f}%."
        elif optimization_goal == 'solubility':
            explanation += f"This should result in {solubility:.1f}% solubility."
        else:
            explanation += (
                f"This should achieve {expr_level:.1f}% expression level "
                f"and {solubility:.1f}% solubility."
            )
            
        return explanation 