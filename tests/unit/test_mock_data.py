"""
Unit tests for the mock data generation module.

This module tests the functionality of the MockProteinExpressionDataGenerator class,
ensuring that it generates valid data within expected ranges and with correct properties.
"""

import pytest
from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator


def test_mock_data_generator() -> None:
    """
    Test the mock data generator functionality.

    Verifies that:
    - The correct number of records are generated
    - All required columns are present
    - Generated values are within expected ranges
    - Data types are correct
    """
    # Test data generation
    generator = MockProteinExpressionDataGenerator(num_records=10)
    df = generator.generate()

    # Check basic properties
    assert len(df) == 10
    assert all(
        col in df.columns
        for col in [
            "experiment_id",
            "host_organism",
            "vector_type",
            "induction_condition",
            "media_type",
            "temperature",
            "induction_time",
            "expression_level",
            "solubility",
        ]
    )

    # Check value ranges
    assert all(20 <= temp <= 37 for temp in df["temperature"])
    assert all(2 <= time <= 24 for time in df["induction_time"])
    assert all(0 <= level <= 100 for level in df["expression_level"])
    assert all(0 <= sol <= 100 for sol in df["solubility"])
