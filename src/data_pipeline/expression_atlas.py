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
    """Client for accessing Expression Atlas data."""
    
    def __init__(self):
        """Initialize the Expression Atlas client with base URL and logging."""
        self.base_url = "https://www.ebi.ac.uk/gxa/api"
        self.logger = logging.getLogger(__name__)
        
    def download_expression_data(self, experiment_id: str) -> pd.DataFrame:
        """
        Download gene expression data from Expression Atlas for a given experiment.
        
        Args:
            experiment_id (str): The Expression Atlas experiment ID (e.g., 'E-MTAB-4045')
            
        Returns:
            pd.DataFrame: A DataFrame containing gene expression data with genes as rows
                        and samples as columns.
                        
        Raises:
            requests.exceptions.RequestException: If there's an error downloading the data
            ValueError: If the data format is unexpected or cannot be parsed
        """
        try:
            # First get experiment type
            metadata_url = f"{self.base_url}/experiments/{experiment_id}"
            self.logger.info(f"Getting experiment metadata from {metadata_url}")
            metadata_response = requests.get(metadata_url)
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            
            # Construct the URL for expression data
            if metadata.get('type') == 'RNASEQ_MRNA_BASELINE':
                url = f"{self.base_url}/baseline/experiments/{experiment_id}/expression"
            else:
                url = f"{self.base_url}/differential/experiments/{experiment_id}/expression"
                
            self.logger.info(f"Downloading expression data from {url}")
            
            # Get expression data
            params = {
                'format': 'tsv',
                'unit': 'TPM'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse TSV data
            df = pd.read_csv(StringIO(response.text), sep='\t')
            
            # Process the data based on experiment type
            if metadata.get('type') == 'RNASEQ_MRNA_BASELINE':
                # For baseline experiments, data is already in TPM format
                df.set_index(['Gene ID', 'Gene Name'], inplace=True)
            else:
                # For differential experiments, we need to extract TPM values
                df = self._process_differential_data(df)
            
            # Convert expression values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.logger.info(f"Successfully downloaded and parsed data for {experiment_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            raise
        except (ValueError, pd.errors.EmptyDataError) as e:
            self.logger.error(f"Error parsing data: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise
            
    def _process_differential_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process differential expression data to extract TPM values."""
        # Extract TPM values from the data
        tpm_cols = [col for col in df.columns if 'TPM' in col]
        if not tpm_cols:
            raise ValueError("No TPM values found in differential expression data")
            
        # Keep only Gene ID, Gene Name, and TPM columns
        result_df = df[['Gene ID', 'Gene Name'] + tpm_cols].copy()
        
        # Clean up column names
        result_df.columns = [col.replace('_TPM', '') for col in result_df.columns]
        
        # Set index
        result_df.set_index(['Gene ID', 'Gene Name'], inplace=True)
        
        return result_df
        
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
        
    def fetch_training_data(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Fetch and combine training data from multiple experiments.
        
        Args:
            experiment_ids: List of experiment accession IDs
            
        Returns:
            Combined DataFrame with protein expression metrics
        """
        all_data = []
        
        for experiment_id in experiment_ids:
            try:
                logger.info(f"Processing experiment {experiment_id}")
                
                # Get expression data
                expression_df = self.download_expression_data(experiment_id)
                
                if expression_df is not None and not expression_df.empty:
                    # Get experiment metadata for conditions
                    metadata = self.get_experiment_metadata(experiment_id)
                    
                    # Process expression data into protein metrics
                    processed_df = self._process_expression_data(expression_df, metadata)
                    
                    # Add experiment ID
                    processed_df['experiment_id'] = experiment_id
                    all_data.append(processed_df)
                else:
                    logger.warning(f"No data found for experiment {experiment_id}")
                    
            except Exception as e:
                logger.error(f"Error processing experiment {experiment_id}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid data found in any of the experiments")
            
        # Combine all processed data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully combined data from {len(all_data)} experiments")
        
        return combined_df
        
    def _process_expression_data(self, df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Process raw expression data into protein expression metrics.
        
        Args:
            df: Raw expression data DataFrame
            metadata: Experiment metadata
            
        Returns:
            DataFrame with protein expression metrics
        """
        # Extract experimental conditions from metadata
        conditions = self._extract_conditions(metadata)
        
        # Calculate protein expression metrics
        metrics_df = pd.DataFrame()
        
        # Calculate expression level (normalized TPM values)
        metrics_df['expression_level'] = df.mean(axis=1).clip(0, 1)  # Normalize to 0-1
        
        # Calculate expression stability (inverse of coefficient of variation)
        metrics_df['expression_stability'] = 1 - (df.std(axis=1) / df.mean(axis=1)).clip(0, 1)
        
        # Calculate solubility prediction (based on sequence properties if available)
        # For now, we'll use expression stability as a proxy
        metrics_df['solubility'] = metrics_df['expression_stability']
        
        # Add experimental conditions
        for condition, value in conditions.items():
            metrics_df[condition] = value
            
        return metrics_df
        
    def _extract_conditions(self, metadata: Dict) -> Dict:
        """
        Extract experimental conditions from metadata.
        
        Args:
            metadata: Experiment metadata
            
        Returns:
            Dictionary of experimental conditions
        """
        conditions = {}
        
        # Extract common experimental conditions
        if 'experimentalConditions' in metadata:
            for condition in metadata['experimentalConditions']:
                name = condition.get('name', '').lower().replace(' ', '_')
                value = condition.get('value', 'unknown')
                conditions[name] = value
                
        # Add default values for missing conditions
        default_conditions = {
            'temperature': 37.0,  # Standard growth temperature
            'induction_time': 4.0,  # Standard induction time
            'host_organism': 'E. coli',  # Most common host
            'vector_type': 'unknown',
            'induction_condition': 'unknown',
            'media_type': 'LB'  # Standard media
        }
        
        for condition, default_value in default_conditions.items():
            if condition not in conditions:
                conditions[condition] = default_value
                
        return conditions

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