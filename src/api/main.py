from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
import logging

from src.data_pipeline.expression_data_processor import ExpressionDataProcessor
from src.ml_models.predictor import ProteinExpressionPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Expression Optimization API")

# Initialize components
processor = ExpressionDataProcessor()
predictor = ProteinExpressionPredictor()

# List of experiments to use for training
TRAINING_EXPERIMENTS = [
    'E-MTAB-4045',  # Arabidopsis thaliana development
    'E-MTAB-5214',  # Human cell lines
    'E-MTAB-5432',  # Mouse tissues
]

# Load and process training data
try:
    train_data = processor.get_training_data(TRAINING_EXPERIMENTS)
    predictor.train(train_data)
    logger.info("Successfully loaded training data and trained model")
except Exception as e:
    logger.error(f"Error loading training data: {str(e)}")
    train_data = pd.DataFrame()  # Empty DataFrame as fallback


class ProteinExpressionRequest(BaseModel):
    """Request model for protein expression prediction.

    Attributes:
        host_organism (str): The organism used for protein expression (e.g., E. coli)
        vector_type (str): The type of expression vector used
        induction_condition (str): The condition used for protein induction
        media_type (str): The type of growth media used
        temperature (float): The temperature used for expression
        induction_time (float): The time duration for induction
        description (Optional[str]): Optional description of the experiment
    """

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
    """Response model for protein expression prediction.

    Attributes:
        predicted_expression_level (float): Predicted expression level (0-100)
        predicted_solubility (float): Predicted solubility (0-100)
        feature_importance (Dict[str, float]): Importance scores for each feature
        timestamp (datetime): When the prediction was made
    """

    predicted_expression_level: float
    predicted_solubility: float
    feature_importance: Dict[str, float]
    timestamp: datetime


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint that returns a welcome message.

    Returns:
        Dict[str, str]: A welcome message for the API
    """
    return {"message": "Welcome to Protein Expression Optimization API"}


@app.get("/valid-categories")
async def get_valid_categories() -> Dict[str, List[str]]:
    """Get the list of valid categories for each field.

    Returns:
        Dict[str, List[str]]: Dictionary containing valid options for each field
    """
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
async def predict_expression(request: ProteinExpressionRequest) -> PredictionResponse:
    """Predict protein expression level and solubility.

    Args:
        request: The prediction request containing experimental parameters

    Returns:
        PredictionResponse containing predicted values and feature importance
    """
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Make prediction
        predictions = predictor.predict(input_data)

        # Get feature importance
        importance = predictor.get_feature_importance()

        return PredictionResponse(
            predicted_expression_level=float(predictions[0][0]),
            predicted_solubility=float(predictions[0][1]),
            feature_importance=importance,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}",
        )


@app.get("/experiments")
async def get_available_experiments() -> List[Dict[str, Any]]:
    """Get list of available experiments for training.

    Returns:
        List of experiment metadata
    """
    try:
        experiments = []
        for exp_id in TRAINING_EXPERIMENTS:
            try:
                metadata = processor.atlas_client.get_experiment_metadata(exp_id)
                experiments.append({
                    'id': exp_id,
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'species': metadata.get('species', ''),
                    'experiment_type': metadata.get('experiment_type', '')
                })
            except Exception as e:
                logger.error(f"Error getting metadata for experiment {exp_id}: {str(e)}")
                continue
                
        return experiments
    except Exception as e:
        logger.error(f"Error getting experiments: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting experiments: {str(e)}",
        )


@app.get("/generate-sample")
async def generate_sample() -> Dict[str, Any]:
    """Generate a sample experiment from the training data.

    Returns:
        Dict[str, Any]: A sample experiment with all required fields
    """
    try:
        if train_data.empty:
            raise HTTPException(
                status_code=500,
                detail="No training data available"
            )
            
        # Get a random sample from the training data
        sample = train_data.sample(n=1).iloc[0]
        
        return {
            "host_organism": sample["host_organism"],
            "vector_type": sample["vector_type"],
            "induction_condition": sample["induction_condition"],
            "media_type": sample["media_type"],
            "temperature": float(sample["temperature"]),
            "induction_time": float(sample["induction_time"]),
            "description": f"Expression of {sample['gene_name']} in {sample['host_organism']}",
            "expression_level": float(sample["expression_level"]),
            "solubility": float(sample["solubility"]),
        }
    except Exception as e:
        logger.error(f"Error generating sample: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating sample: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
