from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
import logging

from src.data_pipeline.mock_data import MockProteinExpressionDataGenerator
from src.ml_models.predictor import ProteinExpressionPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Expression Optimization API")

# Initialize components
generator = MockProteinExpressionDataGenerator(num_records=1000)
train_data = generator.generate()
predictor = ProteinExpressionPredictor()
predictor.train(train_data)


class ProteinExpressionRequest(BaseModel):
    host_organism: str
    vector_type: str
    induction_condition: str
    media_type: str
    temperature: float
    induction_time: float
    description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "host_organism": "E. coli",
                "vector_type": "pET",
                "induction_condition": "IPTG",
                "media_type": "LB",
                "temperature": 37.0,
                "induction_time": 4.0,
                "description": "Expression of GFP in E. coli",
            }
        }


class PredictionResponse(BaseModel):
    predicted_expression_level: float
    predicted_solubility: float
    feature_importance: Dict[str, float]


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to Protein Expression Optimization API"}


@app.get("/valid-categories")
async def get_valid_categories() -> Dict[str, List[str]]:
    """Get the list of valid categories for each field."""
    return {
        "host_organism": ["E. coli", "S. cerevisiae", "P. pastoris", "HEK293", "CHO"],
        "vector_type": ["pET", "pGEX", "pMAL", "pTrc", "pBAD"],
        "induction_condition": [
            "IPTG",
            "Arabinose",
            "Methanol",
            "Galactose",
            "Tetracycline",
        ],
        "media_type": ["LB", "TB", "M9", "YPD", "CD-CHO"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_expression(
    experiment: ProteinExpressionRequest,
) -> PredictionResponse:
    try:
        # Log the received data
        logger.info(f"Received experiment request: {experiment.dict()}")

        # Convert experiment request to DataFrame
        df = pd.DataFrame([experiment.dict()])

        # Validate required columns
        required_columns = [
            "host_organism",
            "vector_type",
            "induction_condition",
            "media_type",
            "temperature",
            "induction_time",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Make prediction
        prediction = predictor.predict(df)[0]

        # Get feature importance
        importance = predictor.get_feature_importance()

        return PredictionResponse(
            predicted_expression_level=float(prediction[0]),
            predicted_solubility=float(prediction[1]),
            feature_importance=importance,
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate-sample")
async def generate_sample() -> Dict[str, Any]:
    """Generate a sample experiment for testing."""
    sample = generator.generate(num_records=1).iloc[0]
    return {
        "host_organism": sample["host_organism"],
        "vector_type": sample["vector_type"],
        "induction_condition": sample["induction_condition"],
        "media_type": sample["media_type"],
        "temperature": sample["temperature"],
        "induction_time": sample["induction_time"],
        "description": sample["description"],
        "expression_level": sample["expression_level"],
        "solubility": sample["solubility"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
