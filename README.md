# Protein Expression Optimizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/keltoumboukra/protein-optimizer/graph/badge.svg?token=AS4ZV2WHT1)](https://codecov.io/gh/keltoumboukra/protein-optimizer)

A data-driven platform that helps researchers predict the presence of post-translational modification (PTM) sites—specifically, N-linked glycosylation sites—in proteins using real data from the EBI/UniProt Proteins API. The pipeline extracts biologically relevant features from protein annotations and sequence, enabling supervised machine learning for PTM site prediction. The project is designed to be extensible, with future plans to incorporate AlphaFold structural features.

![Protein Expression Optimization System Architecture](./assets/system_architecture.png)

## What's Inside

```
protein_optimizer/
├── dashboard/           # Streamlit-based visualization interface
│   ├── app.py          # Main dashboard application with plotly visualizations
│   └── prediction_history.py # Prediction history management
├── src/
│   ├── api/            # FastAPI backend service
│   │   └── main.py     # API endpoints and server configuration
│   ├── data_pipeline/  # Data collection and processing
│   │   ├── collector.py # Real data collection from EBI/UniProt
│   │   ├── processor.py # Data processing for ML
│   │   └── test_ptm_pipeline.py # End-to-end PTM prediction pipeline test
│   └── ml_models/      # Prediction models
│       └── predictor.py # Model code
├── tests/              # Comprehensive test suite
│   ├── integration/    # Integration tests
│   ├── unit/          # Unit tests
│   └── test_prediction_history.py # Prediction history tests
├── data/              # Data storage directory
├── assets/            # Static assets (images, etc.)
├── setup.py           # Package configuration and dependencies
├── requirements.txt   # Development dependencies
├── pytest.ini        # Test configuration settings
├── mypy.ini          # Type checking configuration
└── pyproject.toml    # Code formatting and tool settings
```

## Key Features

- **PTM Site Prediction**: Uses RandomForest models to predict the presence of N-linked glycosylation sites (PTM type `CARBOHYD`) in proteins
- **Real Data Only**: All features and labels are derived from real protein data fetched from the EBI Proteins API. No mock or synthetic data is used.
- **Interactive Dashboard**: Real-time visualization of protein features and predictions
- **REST API**: Easy integration with existing lab workflows
- **Prediction History**: Track and analyze past predictions

## Features Used for Prediction
The following features are extracted for each protein:
- `sequence_length`: Number of amino acids in the protein sequence
- `molecular_weight`: Protein molecular mass (from UniProt annotation)
- `n_cysteines`: Number of cysteine residues in the sequence
- `n_domains`: Number of annotated domains
- `n_disulfide_bonds`: Number of annotated disulfide bonds
- `has_signal_peptide`: 1 if a signal peptide is annotated, else 0
- `has_transmem`: 1 if a transmembrane region is annotated, else 0
- `protein_existence`: Evidence level for protein existence
- `taxonomy_id`: NCBI taxonomy ID for the organism
- `ptm_label`: 1 if at least one N-linked glycosylation site is annotated, else 0

## Getting Started

1. Set up your environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```

This will install all required packages including:
- Core dependencies (pandas, numpy, scikit-learn, etc.)
- API dependencies (FastAPI, uvicorn)
- Visualization dependencies (streamlit, plotly)
- Development tools (pytest, black, mypy)

3. Start the prediction service:
```bash
uvicorn src.api.main:app --reload
```

4. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at http://localhost:8501 and the API at http://localhost:8000.

## Data Pipeline
1. **Data Collection**: 
   - `src/data_pipeline/collector.py` loads `assets/protein_details.json` and extracts features and PTM labels for each protein.
2. **Data Processing**: 
   - `src/data_pipeline/processor.py` cleans the data and splits it into train/test sets for ML.
3. **Model Training & Testing**: 
   - `src/data_pipeline/test_ptm_pipeline.py` runs the full pipeline, trains a random forest classifier, and prints evaluation metrics.

## How to Test the PTM Prediction Pipeline
1. Ensure you have a populated `assets/protein_details.json` file with real protein data from the EBI Proteins API.
2. From the `src/data_pipeline` directory, run:
   ```bash
   python test_ptm_pipeline.py
   ```
3. The script will print:
   - Number of proteins, positives/negatives for PTM
   - Train/test set sizes
   - Classification report (precision, recall, F1, etc.)

## Development Guide

The project uses several development tools to ensure code quality:

### Testing
Run the test suite with coverage report:
```bash
pytest --cov=src tests/
```

### Code Formatting
Format your code using black:
```bash
black .
```

### Type Checking
Check type annotations with mypy:
```bash
mypy src/
```

## Future Roadmap

I'm planning to integrate with key bioinformatics resources:
- AlphaFold for structure prediction
- UniProt for protein properties
- PDB for structural data
- ESM-2 for sequence analysis
- BRENDA for enzyme data
- KEGG for pathway information
- **Support for Additional PTM Types**: Extend the pipeline to other PTMs (e.g., phosphorylation, acetylation).
- **Web Dashboard**: Visualize predictions and protein features interactively.

## Contributing

I welcome contributions! Please read my [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Development workflow
- Code quality standards
- Testing requirements
- Pull request process
- Documentation guidelines

Feel free to open issues or submit pull requests that improve prediction accuracy, add new features, or improve the user interface.

## License

Copyright (c) 2025 Keltoum Boukra

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
