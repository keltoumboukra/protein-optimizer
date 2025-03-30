from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime

from src.data_pipeline.mock_data import MockBugDataGenerator
from src.ml_models.predictor import BugPredictor

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

class PredictionResponse(BaseModel):
    predicted_resolution_time: float
    feature_importance: dict

@app.get("/")
async def root():
    return {"message": "Welcome to BugBrain API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_bug(bug: BugReport):
    try:
        # Convert bug report to DataFrame
        df = pd.DataFrame([bug.dict()])
        
        # Make prediction
        prediction = predictor.predict(df)[0]
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        return PredictionResponse(
            predicted_resolution_time=float(prediction),
            feature_importance=importance
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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