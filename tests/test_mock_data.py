import pytest
from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator


def test_mock_data_generator():
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
