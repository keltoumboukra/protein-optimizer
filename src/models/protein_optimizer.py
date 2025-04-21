import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ProteinOptimizer:
    def __init__(self):
        self.expression_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.solubility_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the optimization models using the provided training data.
        
        Args:
            training_data: DataFrame containing training examples with features
                         and target variables (expression_level, solubility)
        """
        try:
            # Prepare features
            feature_columns = [
                'host_organism', 'vector_type', 'induction_condition',
                'media_type', 'temperature', 'induction_time'
            ]
            X = pd.get_dummies(training_data[feature_columns])
            
            # Train expression model
            y_expression = training_data['expression_level']
            self.expression_model.fit(X, y_expression)
            
            # Train solubility model
            y_solubility = training_data['solubility']
            self.solubility_model.fit(X, y_solubility)
            
            self.is_trained = True
            logger.info("Successfully trained optimization models")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
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