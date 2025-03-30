# Protein Expression Optimization System

ML-Powered Protein Expression Optimization Platform - A comprehensive system for predicting and optimizing protein expression conditions.

## Project Structure

```
protein_optimizer/
├── dashboard/          # Streamlit dashboard
├── data/              # Mock data and data processing
├── src/
│   ├── api/           # FastAPI backend
│   ├── data_pipeline/ # Data processing pipeline
│   └── ml_models/     # ML models for prediction
└── tests/             # Test suite
```

## Features

- Mock data generation for protein expression experiments
- ML model for predicting optimal expression conditions
- Interactive dashboard for visualization
- REST API for model inference
- Test suite for key components

## Future Expansion Possibilities

- Integration with AlphaFold for structure prediction
- Connection to UniProt database for protein properties
- Integration with PDB database for structural data
- Use of ESM-2 or similar models for sequence analysis
- Integration with BRENDA database for enzyme information
- Connection to KEGG database for pathway analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn src.api.main:app --reload
```

4. Start the dashboard:
```bash
streamlit run dashboard/app.py
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check types: `mypy .`
