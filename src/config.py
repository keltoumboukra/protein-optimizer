from typing import Dict, List

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True

# Data Pipeline Configuration
DATA_DIR = "data"
DEFAULT_EXPERIMENTS = [
    "E-MTAB-4045",
    "E-MTAB-4046"
]

# Model Configuration
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

# Optimization Parameters
DEFAULT_PARAM_RANGES = {
    "temperature": {
        "min": 20,
        "max": 42,
        "step": 1
    },
    "induction_time": {
        "min": 1,
        "max": 24,
        "step": 1
    }
}

DEFAULT_CATEGORICAL_OPTIONS = {
    "host_organism": [
        "E. coli BL21",
        "E. coli DH5α",
        "P. pastoris",
        "S. cerevisiae"
    ],
    "vector_type": [
        "pET",
        "pGEX",
        "pMAL",
        "pTrc"
    ],
    "induction_condition": [
        "IPTG",
        "Arabinose",
        "Methanol",
        "Galactose"
    ],
    "media_type": [
        "LB",
        "TB",
        "M9",
        "YPD"
    ]
}

# Feature Configuration
NUMERICAL_FEATURES = [
    "temperature",
    "induction_time",
    "expression_level",
    "solubility"
]

CATEGORICAL_FEATURES = [
    "host_organism",
    "vector_type",
    "induction_condition",
    "media_type"
]

DERIVED_FEATURES = [
    "temperature_category",
    "induction_duration",
    "combined_score"
]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "protein_optimizer.log" 