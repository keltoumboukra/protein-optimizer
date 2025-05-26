from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
import logging
import csv
import json
import os

from src.ml_models.predictor import ProteinExpressionPredictor
from src.api.proteins_api import ProteinsAPIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Expression Optimization API")

# Load real protein data
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'protein_details.json'), 'r') as f:
    real_protein_data = json.load(f)

# TODO: Convert real_protein_data to a DataFrame suitable for model training
# This will depend on your model's expected input format
# For now, we will extract a placeholder DataFrame
def extract_training_df(protein_json_list):
    # Example: extract accession and sequence length as features, add dummy labels
    rows = []
    for entry in protein_json_list:
        details = entry['Details']
        sequence = details.get('sequence', {}).get('sequence', '')
        rows.append({
            'accession': entry['Accession'],
            'sequence_length': len(sequence),
            # Add more real features as needed
            'expression_level': None,  # Placeholder, update with real label if available
            'solubility': None         # Placeholder, update with real label if available
        })
    return pd.DataFrame(rows)

train_data = extract_training_df(real_protein_data)
predictor = ProteinExpressionPredictor()
# Only train if real labels are available
if 'expression_level' in train_data and train_data['expression_level'].notnull().any():
    predictor.train(train_data)
else:
    logger.warning('No real expression labels available for training. Model not trained.')


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
        predicted_expression_level (float): The predicted level of protein expression
        predicted_solubility (float): The predicted solubility of the protein
        feature_importance (Dict[str, float]): Dictionary of feature importance scores
    """

    predicted_expression_level: float
    predicted_solubility: float
    feature_importance: Dict[str, float]


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
async def predict_expression(
    experiment: ProteinExpressionRequest,
) -> PredictionResponse:
    """Predict protein expression level and solubility for a given experiment.

    Args:
        experiment (ProteinExpressionRequest): The experiment parameters

    Returns:
        PredictionResponse: The prediction results including expression level and solubility

    Raises:
        HTTPException: If there are validation errors or processing errors
    """
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
    """Generate a sample experiment for testing.

    Returns:
        Dict[str, Any]: A sample experiment with all required fields and example values
    """
    # Instead of generating a sample, return the first real protein entry as an example
    if len(real_protein_data) > 0:
        details = real_protein_data[0]['Details']
        return details
    else:
        return {"error": "No real protein data available."}


def fetch_and_save_protein_details(tsv_path: str, output_json: str):
    client = ProteinsAPIClient()
    results = []
    with open(tsv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            protein = row['Protein Name']
            organism = row['Organism']
            accession = row['UniProtKB Accession'].strip()
            if not accession:
                logger.info(f"Skipping {protein} ({organism}): No accession provided.")
                continue
            logger.info(f"Fetching details for {protein} ({organism}) [{accession}]")
            details = client.get_protein_details(accession)
            if details:
                results.append({
                    'Protein Name': protein,
                    'Organism': organism,
                    'Accession': accession,
                    'Details': details
                })
            else:
                logger.warning(f"No details found for {accession}")
    with open(output_json, 'w') as out:
        json.dump(results, out, indent=2)
    logger.info(f"Saved details for {len(results)} proteins to {output_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Protein batch fetcher and API server")
    parser.add_argument('--fetch', action='store_true', help='Fetch and save protein details from TSV')
    parser.add_argument('--tsv', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'proteins_of_interest.tsv'), help='Path to TSV file')
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'assets', 'protein_details.json'), help='Output JSON file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    args = parser.parse_args()

    if args.fetch:
        fetch_and_save_protein_details(args.tsv, args.out)
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
