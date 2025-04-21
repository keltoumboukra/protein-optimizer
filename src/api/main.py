from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import logging
from src.data_pipeline.expression_data_processor import ExpressionDataProcessor
from src.models.protein_optimizer import ProteinOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor and model
data_processor = ExpressionDataProcessor()
model = ProteinOptimizer()

class OptimizationRequest(BaseModel):
    experiment_ids: List[str]
    target_protein: str
    optimization_goal: str
    constraints: Optional[Dict] = None

class OptimizationResponse(BaseModel):
    recommendations: List[Dict]
    confidence_scores: List[float]
    explanation: str

class PredictionRequest(BaseModel):
    host_organism: str
    vector_type: str
    induction_condition: str
    media_type: str
    temperature: float
    induction_time: float

@app.get("/")
async def root():
    return {"message": "Protein Expression Optimizer API"}

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_expression(request: OptimizationRequest):
    try:
        # Load training data
        logger.info(f"Loading training data from experiments: {request.experiment_ids}")
        training_data = data_processor.get_training_data(request.experiment_ids)
        
        if training_data.empty:
            raise HTTPException(status_code=400, detail="No valid training data found")
            
        # Train model
        logger.info("Training optimization model")
        model.train(training_data)
        
        # Generate recommendations
        logger.info(f"Generating recommendations for {request.target_protein}")
        recommendations = model.optimize(
            target_protein=request.target_protein,
            optimization_goal=request.optimization_goal,
            constraints=request.constraints
        )
        
        return OptimizationResponse(
            recommendations=recommendations['recommendations'],
            confidence_scores=recommendations['confidence_scores'],
            explanation=recommendations['explanation']
        )
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def list_experiments():
    """List available experiments from Expression Atlas."""
    try:
        # This would typically come from a database or configuration
        # For now, return a static list of example experiments
        experiments = [
            {
                "id": "E-MTAB-4045",
                "name": "RNA-seq of human tissues",
                "description": "Expression data across human tissues",
                "species": "Homo sapiens"
            },
            {
                "id": "E-MTAB-5214",
                "name": "Mouse tissue expression",
                "description": "Expression data across mouse tissues",
                "species": "Mus musculus"
            }
        ]
        return {"experiments": experiments}
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-sample")
async def generate_sample():
    """Generate a sample protein expression data point."""
    try:
        sample = {
            "host_organism": "E. coli",
            "vector_type": "pET",
            "induction_condition": "IPTG",
            "media_type": "LB",
            "temperature": 37.0,
            "induction_time": 4.0,
            "expression_level": 0.8,
            "solubility": 0.7
        }
        return sample
    except Exception as e:
        logger.error(f"Error generating sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions for protein expression conditions."""
    try:
        # For now, return mock predictions
        # In a real implementation, this would use the trained model
        mock_predictions = {
            "predicted_expression_level": 0.85,
            "predicted_solubility": 0.75,
            "feature_importance": {
                "temperature": 0.3,
                "induction_time": 0.25,
                "host_organism": 0.2,
                "vector_type": 0.15,
                "induction_condition": 0.1
            }
        }
        return mock_predictions
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
