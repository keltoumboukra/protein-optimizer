"""
Configuration settings for the Expression Atlas integration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
EXPRESSION_ATLAS_API_URL = "https://www.ebi.ac.uk/gxa/api/v1"
EXPRESSION_ATLAS_USER_AGENT = "Protein-Optimizer/1.0"

# Cache Configuration
CACHE_DIR = Path(os.getenv("CACHE_DIR", "data/cache"))
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "86400"))  # 24 hours in seconds

# API Rate Limiting
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # seconds

# Data Processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10000"))

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True) 