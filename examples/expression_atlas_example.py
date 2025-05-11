#!/usr/bin/env python3
"""
Example script demonstrating the usage of the Expression Atlas API.
This script tests each API endpoint separately to ensure proper functionality.
"""

import os
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExpressionAtlasAPITester:
    """Test class for Expression Atlas API endpoints."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the API tester.
        
        Args:
            cache_dir: Optional directory to cache API responses
        """
        self.base_url = "https://www.ebi.ac.uk/gxa/api/v2"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def test_experiments_summary(self) -> List[Dict]:
        """Test the experiments summary endpoint.
        
        Returns:
            List of experiment summaries
        """
        try:
            url = f"{self.base_url}/experiments"
            logger.info(f"Testing experiments summary endpoint: {url}")
            
            params = {
                'size': 10,
                'species': 'Homo sapiens'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            experiments = data.get('experiments', [])
            logger.info(f"Successfully retrieved {len(experiments)} experiments")
            
            for exp in experiments[:3]:
                logger.info(f"Experiment: {exp.get('experimentAccession')} - {exp.get('experimentType')}")
            
            return experiments
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing experiments summary: {str(e)}")
            raise
    
    def test_experiment_metadata(self, experiment_id: str) -> Dict:
        """Test the experiment metadata endpoint.
        
        Args:
            experiment_id: The experiment accession ID
            
        Returns:
            Experiment metadata
        """
        try:
            url = f"{self.base_url}/experiments/{experiment_id}"
            logger.info(f"Testing experiment metadata endpoint: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully retrieved metadata for {experiment_id}")
            logger.info(f"Title: {data.get('title')}")
            logger.info(f"Type: {data.get('experimentType')}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing experiment metadata: {str(e)}")
            raise
    
    def test_expression_data(self, experiment_id: str) -> pd.DataFrame:
        """Test the expression data endpoint.
        
        Args:
            experiment_id: The experiment accession ID
            
        Returns:
            DataFrame containing expression data
        """
        try:
            metadata = self.test_experiment_metadata(experiment_id)
            
            if metadata.get('type') == 'RNASEQ_MRNA_BASELINE':
                url = f"{self.base_url}/baseline/experiments/{experiment_id}/expression"
            else:
                url = f"{self.base_url}/differential/experiments/{experiment_id}/expression"
            
            logger.info(f"Testing expression data endpoint: {url}")
            
            params = {
                'format': 'tsv',
                'unit': 'TPM'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            df = pd.read_csv(pd.StringIO(response.text), sep='\t')
            logger.info(f"Successfully retrieved expression data with shape: {df.shape}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error accessing expression data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing expression data: {str(e)}")
            raise

def main():
    cache_dir = Path("cache/expression_atlas")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    tester = ExpressionAtlasAPITester(cache_dir=cache_dir)
    
    try:
        logger.info("\n=== Testing Experiments Summary ===")
        experiments = tester.test_experiments_summary()
        
        if not experiments:
            logger.error("No experiments found")
            return
        
        logger.info("\n=== Testing Experiment Metadata ===")
        first_experiment = experiments[0]
        experiment_id = first_experiment.get('experimentAccession')
        
        if not experiment_id:
            logger.error("Could not find experiment ID")
            return
        
        metadata = tester.test_experiment_metadata(experiment_id)
        
        logger.info("\n=== Testing Expression Data ===")
        expression_data = tester.test_expression_data(experiment_id)
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "experiments_summary.json", "w") as f:
            json.dump(experiments, f, indent=2)
        
        with open(output_dir / f"{experiment_id}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        expression_data.to_csv(output_dir / f"{experiment_id}_expression.csv", index=False)
        
        logger.info("\nAll tests completed successfully!")
        logger.info(f"Results saved in {output_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    main() 