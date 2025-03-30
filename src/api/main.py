from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import logging

from src.data_pipeline.mock_data import MockBugDataGenerator
from src.ml_models.predictor import BugPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BugBrain API")

# Initialize components
generator = MockBugDataGenerator(num_records=1000)
train_data = generator.generate()
predictor = BugPredictor()
predictor.train(train_data)

class BugReport(BaseModel):
    instrument: str
    problem_type: str
    severity: str
    status: str
    description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "instrument": "Hamilton",
                "problem_type": "Hardware",
                "severity": "Low",
                "status": "Open",
                "description": "Optional description"
            }
        }

class PredictionResponse(BaseModel):
    predicted_resolution_time: float
    feature_importance: dict

@app.get("/")
async def root():
    return {"message": "Welcome to BugBrain API"}

@app.get("/valid-categories")
async def get_valid_categories():
    """Get the list of valid categories for each field."""
    return {
        "instrument": ["Hamilton", "Tecan", "Beckman", "Agilent", "PerkinElmer"],
        "problem_type": ["Hardware", "Software", "Calibration", "Sample Processing", "Communication"],
        "severity": ["Low", "Medium", "High", "Critical"],
        "status": ["Open", "In Progress", "Resolved", "Closed"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_bug(bug: BugReport):
    try:
        # Log the received data
        logger.info(f"Received bug report: {bug.dict()}")
        
        # Convert bug report to DataFrame
        df = pd.DataFrame([bug.dict()])
        
        # Validate required columns
        required_columns = ["instrument", "problem_type", "severity", "status"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Make prediction
        prediction = predictor.predict(df)[0]
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        return PredictionResponse(
            predicted_resolution_time=float(prediction),
            feature_importance=importance
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-sample")
async def generate_sample():
    """Generate a sample bug report for testing."""
    sample = generator.generate(num_records=1).iloc[0]
    return {
        "instrument": sample["instrument"],
        "problem_type": sample["problem_type"],
        "severity": sample["severity"],
        "status": sample["status"],
        "description": sample["description"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 