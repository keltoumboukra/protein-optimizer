import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional
from src.config import (
    DATA_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES
)
from src.data_pipeline.expression_atlas import ExpressionAtlasClient

class DataProcessor:
    def __init__(self, data_dir: str = DATA_DIR):
        """Initialize the DataProcessor.
        
        Args:
            data_dir (str): Directory to store and load data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.atlas_client = ExpressionAtlasClient()

    def load_training_data(self, experiment_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and combine training data from multiple experiments.
        
        Args:
            experiment_ids (List[str], optional): List of Expression Atlas experiment IDs.
                                                If None, uses default set of experiments.
        
        Returns:
            pd.DataFrame: Combined and processed training data
        """
        try:
            if experiment_ids is None:
                # Use a default set of protein expression experiments
                experiment_ids = [
                    'E-MTAB-4045',  # Example protein expression study
                    'E-MTAB-5214',  # Another example study
                ]
            
            # Fetch data from Expression Atlas
            raw_data = self.atlas_client.fetch_training_data(experiment_ids)
            
            # Process the data
            processed_data = self._process_experiment_data(raw_data)
            
            # Save processed data for future use
            self.save_processed_data(processed_data, 'processed_training_data.csv')
            
            self.logger.info(f"Successfully loaded and processed {len(experiment_ids)} experiments")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            raise

    def _process_experiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process experiment data.
        
        Args:
            df (pd.DataFrame): Raw experiment data
        
        Returns:
            pd.DataFrame: Processed data
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].fillna('unknown')
        
        # Add derived features
        df = self._add_derived_features(df)
        
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with added derived features
        """
        # Temperature categories (if temperature data is available)
        if 'temperature' in df.columns:
            df['temperature_category'] = pd.cut(
                df['temperature'],
                bins=[0, 25, 30, 37, float('inf')],
                labels=['low', 'medium-low', 'medium-high', 'high']
            )
        
        # Induction duration categories (if induction time is available)
        if 'induction_time' in df.columns:
            df['induction_duration'] = pd.cut(
                df['induction_time'],
                bins=[0, 2, 4, 8, float('inf')],
                labels=['very_short', 'short', 'medium', 'long']
            )
        
        # Combined score (if both metrics are available)
        if 'expression_level' in df.columns and 'solubility' in df.columns:
            df['combined_score'] = (df['expression_level'] + df['solubility']) / 2
        
        return df

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data to a file.
        
        Args:
            df (pd.DataFrame): Processed data to save
            filename (str): Name of the output file
        """
        try:
            output_path = self.data_dir / filename
            df.to_csv(output_path, index=False)
            self.logger.info(f"Successfully saved processed data to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise 