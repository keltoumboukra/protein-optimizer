#!/usr/bin/env python3
"""
Script to train the protein optimization model using real Expression Atlas data.
"""

import logging
from pathlib import Path
from src.data_pipeline.data_processor import DataProcessor
from src.models.protein_optimizer import ProteinOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize data processor
        logger.info("Initializing data processor...")
        data_processor = DataProcessor()
        
        # Define experiment IDs to use
        experiment_ids = [
            'E-MTAB-3358',  # RNA-seq of protein coding genes in S. cerevisiae
            'E-GEOD-21520',  # Protein expression profiling in E. coli
            'E-GEOD-59044',  # Protein expression in different growth conditions
        ]
        
        # Load training data from Expression Atlas
        logger.info("Loading training data from Expression Atlas...")
        logger.info(f"Using experiments: {', '.join(experiment_ids)}")
        training_data = data_processor.load_training_data(experiment_ids)
        logger.info(f"Loaded {len(training_data)} samples")
        
        # Print data information
        logger.info("\nData columns:")
        for col in training_data.columns:
            logger.info(f"- {col}")
            
        # Print value ranges for key metrics
        logger.info("\nValue ranges for key metrics:")
        for metric in ['expression_level', 'solubility']:
            if metric in training_data.columns:
                min_val = training_data[metric].min()
                max_val = training_data[metric].max()
                mean_val = training_data[metric].mean()
                logger.info(f"{metric}: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")
        
        # Initialize and train the model
        logger.info("\nInitializing protein optimizer model...")
        optimizer = ProteinOptimizer()
        
        # Train the model
        logger.info("Training model...")
        optimizer.train(
            data=training_data,
            target_columns=['expression_level', 'solubility']
        )
        
        # Get feature importance
        importance = optimizer.get_feature_importance()
        logger.info("\nFeature importance:")
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{feature}: {score:.3f}")
        
        logger.info("\nTraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 