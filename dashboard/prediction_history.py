"""
Prediction history management for the Streamlit dashboard.

This module provides functionality for:
- Saving protein expression predictions
- Loading prediction history
- Comparing predictions
- Exporting prediction data
"""

import pandas as pd
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
import streamlit as st


class PredictionHistory:
    """Manages the history of protein expression predictions."""

    def __init__(self, storage_path: str = "data/predictions"):
        """
        Initialize the prediction history manager.

        Args:
            storage_path: Directory to store prediction history files
        """
        self.storage_path = storage_path
        self._ensure_storage_directory()
        self._counter = 0

    def _ensure_storage_directory(self) -> None:
        """Create the storage directory if it doesn't exist."""
        os.makedirs(self.storage_path, exist_ok=True)

    def save_prediction(self, prediction_data: Dict) -> None:
        """
        Save a prediction to the history.

        Args:
            prediction_data: Dictionary containing prediction details
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._counter += 1
        filename = f"prediction_{timestamp}_{self._counter}.json"
        filepath = os.path.join(self.storage_path, filename)

        # Add timestamp to prediction data
        prediction_data = prediction_data.copy()  # Create a copy to avoid modifying the original
        prediction_data["timestamp"] = timestamp

        with open(filepath, "w") as f:
            json.dump(prediction_data, f, indent=2)

    def load_history(self) -> pd.DataFrame:
        """
        Load all predictions from history.

        Returns:
            DataFrame containing all predictions
        """
        predictions = []
        if os.path.exists(self.storage_path):
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_path, filename)
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            
                            # Handle both single prediction dictionaries and arrays of predictions
                            if isinstance(data, dict):
                                # Single prediction
                                predictions.append(data)
                            elif isinstance(data, list):
                                # Array of predictions
                                for prediction in data:
                                    if isinstance(prediction, dict):
                                        predictions.append(prediction)
                                    else:
                                        st.warning(f"Invalid prediction format in {filename}: item is not a dictionary")
                            else:
                                st.warning(f"Invalid prediction format in {filename}: not a dictionary or array")
                    except (json.JSONDecodeError, Exception) as e:
                        st.error(f"Error loading {filename}: {str(e)}")
                        continue

        if not predictions:
            return pd.DataFrame()

        try:
            return pd.DataFrame.from_records(predictions)
        except Exception as e:
            st.error(f"Error creating DataFrame: {str(e)}")
            return pd.DataFrame()

    def export_history(self, format: str = "csv") -> Optional[str]:
        """
        Export prediction history to a file.

        Args:
            format: Export format ("csv" or "json")

        Returns:
            Path to the exported file
        """
        df = self.load_history()
        if df.empty:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "csv":
            filename = f"prediction_history_{timestamp}.csv"
            filepath = os.path.join(self.storage_path, filename)
            df.to_csv(filepath, index=False)
        else:
            filename = f"prediction_history_{timestamp}.json"
            filepath = os.path.join(self.storage_path, filename)
            df.to_json(filepath, orient="records", indent=2)

        return filepath
