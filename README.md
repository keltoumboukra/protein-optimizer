# Protein Expression Optimization System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A data-driven platform that helps researchers optimize protein expression conditions using machine learning. This tool predicts expression levels and solubility based on experimental parameters, helping streamline the protein production process.

![Protein Expression Optimization System Architecture](./assets/system_architecture.png)

## What's Inside

```
protein_optimizer/
├── dashboard/          # Streamlit-based visualization interface
├── src/
│   ├── api/           # FastAPI backend service
│   ├── data_pipeline/ # Data generation and processing
│   └── ml_models/     # Expression prediction models
└── tests/             # Test coverage
```

## Key Features

- **Smart Predictions**: Uses RandomForest models to predict protein expression levels and solubility
- **Interactive Dashboard**: Real-time visualization of expression data and predictions
- **REST API**: Easy integration with existing lab workflows
- **Rapid Prototyping**: Built-in mock data generation for testing and development

## Getting Started

1. Set up your environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Start the prediction service:
```bash
uvicorn src.api.main:app --reload
```

3. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

## Development Guide

```bash
# Run the test suite
pytest

# Format your code
black .

# Check typing
mypy .
```

## Future Roadmap

I'm planning to integrate with key bioinformatics resources:
- AlphaFold for structure prediction
- UniProt for protein properties
- PDB for structural data
- ESM-2 for sequence analysis
- BRENDA for enzyme data
- KEGG for pathway information

## Contributing

I welcome contributions! Feel free to open issues or submit pull requests that improve prediction accuracy, add new features, or enhance the user interface.

## License

MIT License - feel free to use and modify as needed.
