import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

class MockBugDataGenerator:
    def __init__(self, num_records: int = 100):
        self.num_records = num_records
        self.instruments = [
            "Hamilton", "Tecan", "Beckman", "Agilent", "PerkinElmer"
        ]
        self.problem_types = [
            "Hardware", "Software", "Calibration", "Sample Processing", "Communication"
        ]
        self.severity_levels = ["Low", "Medium", "High", "Critical"]
        self.statuses = ["Open", "In Progress", "Resolved", "Closed"]
        
    def generate(self, num_records: int = None) -> pd.DataFrame:
        """Generate mock bug data."""
        if num_records is None:
            num_records = self.num_records
            
        data = {
            "ticket_id": [f"BUG-{i:04d}" for i in range(num_records)],
            "created_date": self._generate_dates(num_records),
            "instrument": np.random.choice(self.instruments, num_records),
            "problem_type": np.random.choice(self.problem_types, num_records),
            "severity": np.random.choice(self.severity_levels, num_records),
            "status": np.random.choice(self.statuses, num_records),
            "resolution_time_hours": np.random.exponential(24, num_records),
            "description": [self._generate_description() for _ in range(num_records)],
            "solution": [self._generate_solution() for _ in range(num_records)]
        }
        
        df = pd.DataFrame(data)
        df["created_date"] = pd.to_datetime(df["created_date"])
        return df
    
    def _generate_dates(self, num_records: int) -> List[datetime]:
        """Generate random dates within the last 6 months."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        return [
            start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            for _ in range(num_records)
        ]
    
    def _generate_description(self) -> str:
        """Generate a realistic bug description."""
        templates = [
            "Instrument {instrument} failed during {operation} with error code {code}",
            "Calibration issues detected on {instrument} during {operation}",
            "Communication timeout between {instrument} and {component}",
            "Sample processing error on {instrument}: {error_type}"
        ]
        
        template = np.random.choice(templates)
        operations = ["sample loading", "pipetting", "washing", "incubation", "reading"]
        error_codes = ["E001", "E002", "E003", "E004", "E005"]
        components = ["deck", "tip rack", "plate reader", "washer", "incubator"]
        error_types = ["volume mismatch", "position error", "timing issue", "sensor failure"]
        
        return template.format(
            instrument=np.random.choice(self.instruments),
            operation=np.random.choice(operations),
            code=np.random.choice(error_codes),
            component=np.random.choice(components),
            error_type=np.random.choice(error_types)
        )
    
    def _generate_solution(self) -> str:
        """Generate a realistic solution description."""
        templates = [
            "Performed {action} on {component}",
            "Updated {component} firmware to version {version}",
            "Replaced {component} and recalibrated",
            "Adjusted {parameter} settings to {value}"
        ]
        
        template = np.random.choice(templates)
        actions = ["calibration", "maintenance", "cleaning", "reboot", "replacement"]
        components = ["pipette", "sensor", "motor", "controller", "reader"]
        versions = ["2.1.0", "2.2.0", "2.3.0", "2.4.0"]
        parameters = ["speed", "temperature", "pressure", "volume", "timing"]
        values = ["optimal", "recommended", "standard", "default"]
        
        return template.format(
            action=np.random.choice(actions),
            component=np.random.choice(components),
            version=np.random.choice(versions),
            parameter=np.random.choice(parameters),
            value=np.random.choice(values)
        )

if __name__ == "__main__":
    # Example usage
    generator = MockBugDataGenerator(num_records=10)
    df = generator.generate()
    print(df.head()) 