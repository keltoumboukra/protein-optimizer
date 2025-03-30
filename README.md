# BugBrain

ML-Powered Bug Analysis System - Toy Project

A simplified version of the bug analysis system for practicing key components and workflows.

## Project Structure

```
bugbrain/
├── dashboard/          # Streamlit dashboard
├── data/              # Mock data and data processing
├── src/
│   ├── api/           # FastAPI backend
│   ├── data_pipeline/ # Data processing pipeline
│   └── ml_models/     # ML models for prediction
└── tests/             # Test suite
```

## Features

- Mock data generation for lab automation bugs
- Basic ML model for bug prediction
- Interactive dashboard for visualization
- REST API for model inference
- Test suite for key components

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
