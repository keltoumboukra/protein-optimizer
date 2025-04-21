"""
Expression Atlas data integration module.

This module provides functionality to fetch and process protein expression data
from the EMBL-EBI Expression Atlas database.
"""

import requests
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from datetime import datetime
import logging
import json
import os
import time
from pathlib import Path
from io import StringIO

logger = logging.getLogger(__name__)

class ExpressionAtlasClient:
    """Client for interacting with the Expression Atlas API."""
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the Expression Atlas client.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.base_url = "https://www.ebi.ac.uk/gxa/api"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """
        Get a cached API response if available.
        
        Args:
            cache_key: Unique key for the cached response
            
        Returns:
            Cached response or None if not available
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache file: {str(e)}")
        return None
    
    def _cache_response(self, cache_key: str, response_data: Dict) -> None:
        """
        Cache an API response.
        
        Args:
            cache_key: Unique key for the cached response
            response_data: Response data to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(response_data, f)
        except Exception as e:
            logger.warning(f"Error writing to cache file: {str(e)}")
    
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
        # Create cache key
        cache_key = f"search_{organism}_{experiment_type}_{limit}"
        cached_data = self._get_cached_response(cache_key)
        if cached_data:
            logger.info("Using cached search results")
            return cached_data
        
        # Get all experiments summary
        response = self.session.get(f"{self.BASE_URL}/experiments-summary")
        response.raise_for_status()
        
        data = response.json()
        
        # Filter results based on parameters
        experiments = []
        for exp in data:
            if organism and organism.lower() not in exp.get('species', '').lower():
                continue
            if experiment_type and experiment_type.lower() not in exp.get('experimentType', '').lower():
                continue
            experiments.append(exp)
            
            if len(experiments) >= limit:
                break
        
        # Cache the results
        self._cache_response(cache_key, experiments)
        
        return experiments
    
    def get_experiment_data(self, experiment_accession: str) -> Dict:
        """
        Fetch detailed data for a specific experiment.
        
        Args:
            experiment_accession: The experiment accession ID
            
        Returns:
            Experiment data including expression values and metadata
        """
        # Create cache key
        cache_key = f"experiment_{experiment_accession}"
        cached_data = self._get_cached_response(cache_key)
        if cached_data:
            logger.info(f"Using cached data for experiment {experiment_accession}")
            return cached_data
        
        response = self.session.get(f"{self.BASE_URL}/experiments/{experiment_accession}")
        response.raise_for_status()
        
        data = response.json()
        
        # Cache the results
        self._cache_response(cache_key, data)
        
        return data
    
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
        # Create cache key
        cache_key = f"expression_{experiment_accession}_{gene_id}"
        cached_data = self._get_cached_response(cache_key)
        if cached_data:
            logger.info(f"Using cached expression data for experiment {experiment_accession}")
            return pd.DataFrame(cached_data)
        
        # Get the experiment data
        experiment_data = self.get_experiment_data(experiment_accession)
        
        # Extract expression data
        data = {
            'gene_id': [],
            'expression_level': [],
            'experimental_condition': []
        }
        
        # Process the expression data
        if 'results' in experiment_data:
            results = experiment_data['results']
            if isinstance(results, list):
                for result in results:
                    if 'geneId' in result:
                        data['gene_id'].append(result['geneId'])
                    else:
                        data['gene_id'].append('unknown')
                        
                    if 'expressionLevel' in result:
                        data['expression_level'].append(result['expressionLevel'])
                    else:
                        data['expression_level'].append(0)
                        
                    if 'condition' in result:
                        data['experimental_condition'].append(result['condition'])
                    else:
                        data['experimental_condition'].append('unknown')
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Cache the results
        self._cache_response(cache_key, df.to_dict(orient='records'))
        
        return df
    
    def get_experiment_metadata(self, experiment_id: str) -> Dict[str, Any]:
        """Get metadata for an experiment.
        
        Args:
            experiment_id: The Expression Atlas experiment ID
            
        Returns:
            Dictionary containing experiment metadata
        """
        url = f"{self.base_url}/experiments/{experiment_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
        
    def download_expression_data(self, experiment_id: str) -> pd.DataFrame:
        """Download expression data for an experiment.
        
        Args:
            experiment_id: The Expression Atlas experiment ID
            
        Returns:
            DataFrame containing expression data
        """
        # First get the experiment details to find the correct download URL
        metadata = self.get_experiment_metadata(experiment_id)
        
        # Get the download URL from the metadata
        download_url = None
        for resource in metadata.get('resources', []):
            if resource.get('type') == 'ExperimentDownloadSupplier.RnaSeqBaseline':
                download_url = resource.get('url')
                break
                
        if not download_url:
            raise ValueError(f"No RNA-seq baseline data found for experiment {experiment_id}")
            
        # Download the data
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Parse the TSV data
        df = pd.read_csv(pd.StringIO(response.text), sep='\t')
        
        # Extract gene IDs and expression values
        gene_ids = df['Gene ID'].values
        expression_values = df.iloc[:, 2:].values  # Skip Gene ID and Gene Name columns
        
        # Create DataFrame with genes as rows and samples as columns
        result_df = pd.DataFrame(
            expression_values,
            index=gene_ids,
            columns=df.columns[2:]
        )
        
        return result_df

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
    column_mappings = {
        'geneId': 'gene_id',
        'gene_id': 'gene_id',
        'gene': 'gene_id',
        'expressionLevel': 'expression_level',
        'expression_level': 'expression_level',
        'expression': 'expression_level',
        'condition': 'experimental_condition',
        'experimental_condition': 'experimental_condition'
    }
    
    df = df.rename(columns=column_mappings)
    
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
        logger.error(f"Available columns: {', '.join(df.columns)}")
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