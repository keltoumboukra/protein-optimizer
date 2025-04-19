# Protein Expression Optimization System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/keltoumboukra/protein-optimizer/graph/badge.svg?token=AS4ZV2WHT1)](https://codecov.io/gh/keltoumboukra/protein-optimizer)

## Test Coverage Overview
![Code Coverage Sunburst](https://codecov.io/gh/keltoumboukra/protein-optimizer/graphs/sunburst.svg?token=AS4ZV2WHT1)

A data-driven platform that helps researchers optimize protein expression conditions using machine learning. This tool predicts expression levels and solubility based on experimental parameters, helping streamline the protein production process.

![Protein Expression Optimization System Architecture](./assets/system_architecture.png)

## Project Structure

```
protein_optimizer/
├── dashboard/           # Streamlit visualization interface
├── src/
│   ├── api/            # FastAPI backend service
│   ├── data_pipeline/  # Data processing
│   └── ml_models/      # ML prediction models
├── tests/              # Test suite
└── data/               # Data storage
```

## Key Features

- **ML-Powered Predictions**: RandomForest models for expression and solubility prediction
- **Interactive Dashboard**: Real-time data visualization
- **REST API**: Integration with lab workflows
- **Development Tools**: Comprehensive testing and type checking

## Quick Start

1. Set up environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

2. Start services:
```bash
# Start API (http://localhost:8000)
uvicorn src.api.main:app --reload

# Start Dashboard (http://localhost:8501)
streamlit run dashboard/app.py
```

## Development

```bash
# Run tests with coverage
pytest --cov=src tests/

# Format code
black .

# Type checking
mypy src/
```

## License

MIT License - feel free to use and modify as needed.
