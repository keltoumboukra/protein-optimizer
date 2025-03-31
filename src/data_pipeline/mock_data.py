import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class MockProteinExpressionDataGenerator:
    def __init__(self, num_records: int = 100) -> None:
        self.num_records = num_records
        self.host_organisms: List[str] = [
            "E. coli",
            "S. cerevisiae",
            "P. pastoris",
            "HEK293",
            "CHO",
        ]
        self.vector_types: List[str] = ["pET", "pGEX", "pMAL", "pTrc", "pBAD"]
        self.induction_conditions: List[str] = [
            "IPTG",
            "Arabinose",
            "Methanol",
            "Galactose",
            "Tetracycline",
        ]
        self.media_types: List[str] = ["LB", "TB", "M9", "YPD", "CD-CHO"]

    def generate(self, num_records: Optional[int] = None) -> pd.DataFrame:
        """Generate mock protein expression data."""
        if num_records is None:
            num_records = self.num_records

        data: Dict[str, List] = {
            "experiment_id": [f"EXP-{i:04d}" for i in range(num_records)],
            "date": self._generate_dates(num_records),
            "host_organism": np.random.choice(
                self.host_organisms, num_records
            ).tolist(),
            "vector_type": np.random.choice(self.vector_types, num_records).tolist(),
            "induction_condition": np.random.choice(
                self.induction_conditions, num_records
            ).tolist(),
            "media_type": np.random.choice(self.media_types, num_records).tolist(),
            "temperature": np.random.uniform(20, 37, num_records).tolist(),
            "induction_time": np.random.uniform(2, 24, num_records).tolist(),
            "expression_level": np.random.uniform(0, 100, num_records).tolist(),
            "solubility": np.random.uniform(0, 100, num_records).tolist(),
            "description": [self._generate_description() for _ in range(num_records)],
            "notes": [self._generate_notes() for _ in range(num_records)],
        }

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _generate_dates(self, num_records: int) -> List[datetime]:
        """Generate random dates within the last 6 months."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        return [
            start_date
            + timedelta(
                seconds=np.random.randint(
                    0, int((end_date - start_date).total_seconds())
                )
            )
            for _ in range(num_records)
        ]

    def _generate_description(self) -> str:
        """Generate a realistic experiment description."""
        templates: List[str] = [
            "Expression of {protein} in {host} using {vector} vector",
            "Optimization of {protein} expression in {host} with {media} media",
            "Testing {protein} expression under {condition} induction",
            "Scale-up experiment for {protein} in {host}",
        ]

        template = str(np.random.choice(templates))
        proteins = ["GFP", "His-tag protein", "Fusion protein", "Enzyme", "Antibody"]

        return template.format(
            protein=str(np.random.choice(proteins)),
            host=str(np.random.choice(self.host_organisms)),
            vector=str(np.random.choice(self.vector_types)),
            media=str(np.random.choice(self.media_types)),
            condition=str(np.random.choice(self.induction_conditions)),
        )

    def _generate_notes(self) -> str:
        """Generate realistic experiment notes."""
        templates: List[str] = [
            "Adjusted {parameter} to {value} for better expression",
            "Observed {observation} during induction",
            "Modified {component} protocol to improve {outcome}",
            "Troubleshooting {issue} with {solution}",
        ]

        template = str(np.random.choice(templates))
        parameters = [
            "temperature",
            "induction time",
            "media composition",
            "OD600",
            "pH",
        ]
        values = ["optimal", "recommended", "standard", "default"]
        observations = [
            "aggregation",
            "low yield",
            "high background",
            "good expression",
        ]
        components = ["induction", "harvesting", "lysis", "purification"]
        outcomes = ["yield", "solubility", "activity", "purity"]
        issues = ["low expression", "insolubility", "degradation", "contamination"]
        solutions = [
            "temperature adjustment",
            "media optimization",
            "protocol modification",
        ]

        return template.format(
            parameter=str(np.random.choice(parameters)),
            value=str(np.random.choice(values)),
            observation=str(np.random.choice(observations)),
            component=str(np.random.choice(components)),
            outcome=str(np.random.choice(outcomes)),
            issue=str(np.random.choice(issues)),
            solution=str(np.random.choice(solutions)),
        )


if __name__ == "__main__":
    # Example usage
    generator = MockProteinExpressionDataGenerator(num_records=10)
    df = generator.generate()
    print(df.head())
