"""
Expression Atlas data integration module.

This module provides functionality to fetch and process protein expression data
from the EMBL-EBI Expression Atlas database.
"""

import requests
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExpressionAtlasClient:
    """Client for interacting with the Expression Atlas API."""
    
    BASE_URL = "https://www.ebi.ac.uk/gxa/api/v1"
    
    def __init__(self):
        """Initialize the Expression Atlas client."""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "Protein-Optimizer/1.0"
        })
    
    def search_experiments(self, 
                          organism: Optional[str] = None,
                          experiment_type: Optional[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Search for experiments in Expression Atlas.
        
        Args:
            organism: Filter by organism (e.g., 'Homo sapiens')
            experiment_type: Filter by experiment type (e.g., 'RNA-seq')
            limit: Maximum number of results to return
            
        Returns:
            List of experiment metadata
        """
        params = {
            "limit": limit,
            "offset": 0
        }
        
        if organism:
            params["organism"] = organism
        if experiment_type:
            params["experimentType"] = experiment_type
            
        response = self.session.get(f"{this.BASE_URL}/experiments", params=params)
        response.raise_for_status()
        
        return response.json()["experiments"]
    
    def get_experiment_data(self, experiment_accession: str) -> Dict:
        """
        Fetch detailed data for a specific experiment.
        
        Args:
            experiment_accession: The experiment accession ID
            
        Returns:
            Experiment data including expression values and metadata
        """
        response = self.session.get(
            f"{this.BASE_URL}/experiments/{experiment_accession}"
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_expression_data(self, 
                          experiment_accession: str,
                          gene_id: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch expression data for a specific experiment and optionally filter by gene.
        
        Args:
            experiment_accession: The experiment accession ID
            gene_id: Optional gene ID to filter results
            
        Returns:
            DataFrame containing expression data
        """
        params = {}
        if gene_id:
            params["geneId"] = gene_id
            
        response = this.session.get(
            f"{this.BASE_URL}/experiments/{experiment_accession}/results",
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        return pd.DataFrame(data["results"])
    
    def get_experimental_conditions(self, experiment_accession: str) -> Dict:
        """
        Fetch experimental conditions and metadata for a specific experiment.
        
        Args:
            experiment_accession: The experiment accession ID
            
        Returns:
            Dictionary containing experimental conditions and metadata
        """
        response = this.session.get(
            f"{this.BASE_URL}/experiments/{experiment_accession}/conditions"
        )
        response.raise_for_status()
        
        return response.json()

def process_expression_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw expression data into a standardized format.
    
    Args:
        df: Raw expression data DataFrame
        
    Returns:
        Processed DataFrame with standardized columns
    """
    # Add processing timestamp
    df['processed_at'] = datetime.utcnow()
    
    # Standardize column names
    df = df.rename(columns={
        'geneId': 'gene_id',
        'expressionLevel': 'expression_level',
        'condition': 'experimental_condition'
    })
    
    # Convert expression levels to numeric
    df['expression_level'] = pd.to_numeric(df['expression_level'], errors='coerce')
    
    return df

def validate_expression_data(df: pd.DataFrame) -> bool:
    """
    Validate the processed expression data.
    
    Args:
        df: Processed expression data DataFrame
        
    Returns:
        True if data is valid, False otherwise
    """
    required_columns = ['gene_id', 'expression_level', 'experimental_condition']
    
    # Check for required columns
    if not all(col in df.columns for col in required_columns):
        logger.error("Missing required columns in expression data")
        return False
    
    # Check for null values in critical columns
    if df[required_columns].isnull().any().any():
        logger.error("Found null values in required columns")
        return False
    
    # Check for negative expression levels
    if (df['expression_level'] < 0).any():
        logger.error("Found negative expression levels")
        return False
    
    return True 