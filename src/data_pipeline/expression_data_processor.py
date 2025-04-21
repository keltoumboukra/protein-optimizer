import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .expression_atlas import ExpressionAtlasClient
import logging

logger = logging.getLogger(__name__)

class ExpressionDataProcessor:
    """Processes Expression Atlas data into protein expression format."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the processor with an Expression Atlas client.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.atlas_client = ExpressionAtlasClient(cache_dir=cache_dir)
        
    def get_protein_expression_data(self, experiment_id: str) -> pd.DataFrame:
        """
        Get protein expression data from Expression Atlas and convert to our format.
        
        Args:
            experiment_id: The Expression Atlas experiment ID
            
        Returns:
            DataFrame containing protein expression data in our format
        """
        try:
            # Get raw expression data
            raw_data = self.atlas_client.download_expression_data(experiment_id)
            
            # Get experiment metadata
            metadata = self.atlas_client.get_experiment_metadata(experiment_id)
            
            # Process the data
            processed_data = []
            
            # For each gene (row) in the expression data
            for gene_id, row in raw_data.iterrows():
                # Calculate expression level (normalized to 0-100 scale)
                max_expr = row.max()
                if max_expr > 0:
                    normalized_expr = (row / max_expr) * 100
                else:
                    normalized_expr = row
                    
                # For each experimental condition (column)
                for condition, expr_level in normalized_expr.items():
                    # Create a record for this gene-condition pair
                    record = {
                        'experiment_id': experiment_id,
                        'gene_id': gene_id,
                        'gene_name': gene_id,  # We don't have gene names in the new format
                        'host_organism': metadata.get('species', 'Unknown'),
                        'experimental_condition': condition,
                        'expression_level': expr_level,
                        # Estimate solubility based on expression level
                        'solubility': min(100, expr_level * 1.2),  # Assume some correlation
                        'experiment_type': metadata.get('experimentType', 'Unknown'),
                        'technology_type': metadata.get('technologyType', ['Unknown'])[0],
                        'vector_type': 'Native',  # Default to native expression
                        'induction_condition': 'Endogenous',  # Default to endogenous expression
                        'media_type': 'Standard',  # Default to standard media
                        'temperature': 37.0,  # Default to standard temperature
                        'induction_time': 24.0  # Default to standard induction time
                    }
                    processed_data.append(record)
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
            return df
            
        except Exception as e:
            logger.error(f"Error processing experiment {experiment_id}: {str(e)}")
            raise
    
    def get_training_data(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Get training data from multiple experiments.
        
        Args:
            experiment_ids: List of Expression Atlas experiment IDs
            
        Returns:
            Combined DataFrame containing training data
        """
        all_data = []
        for exp_id in experiment_ids:
            try:
                data = self.get_protein_expression_data(exp_id)
                all_data.append(data)
            except Exception as e:
                logger.error(f"Error processing experiment {exp_id}: {str(e)}")
                continue
                
        if not all_data:
            raise ValueError("No valid data could be processed from the provided experiments")
            
        return pd.concat(all_data, ignore_index=True) 