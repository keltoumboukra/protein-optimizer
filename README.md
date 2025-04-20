# Protein Expression Optimizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/keltoumboukra/protein-optimizer/actions/workflows/ci.yml)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/keltoumboukra/protein-optimizer/graph/badge.svg?token=AS4ZV2WHT1)](https://codecov.io/gh/keltoumboukra/protein-optimizer)

A data-driven system that helps researchers optimize protein expression conditions using machine learning. This tool predicts expression levels and solubility based on experimental parameters, helping streamline the protein production process.

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
│   ├── data_pipeline/  # Data generation and processing
│   │   └── mock_data.py # Mock data generation for development
│   └── ml_models/      # Expression prediction models
│       └── predictor.py # RandomForest-based protein expression predictor
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

- **Smart Predictions**: Uses RandomForest models to predict protein expression levels and solubility
- **Interactive Dashboard**: Real-time visualization of expression data and predictions
- **Prediction History**: Track and analyze past predictions
- **REST API**: Easy integration with existing lab workflows
- **Rapid Prototyping**: Built-in mock data generation for testing and development

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

Note: Every time you open a new terminal or IDE session, you'll need to reactivate the virtual environment:
```bash
# Navigate to the project directory
cd /path/to/protein-optimizer

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# You should see (venv) in your terminal prompt when it's activated
```

3. Start the prediction service:
```bash
uvicorn src.api.main:app --reload
```

4. Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at http://localhost:8501 and the API at http://localhost:8000.

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

### Configuration Files
- `pytest.ini`: Test configuration
- `mypy.ini`: Type checking settings
- `pyproject.toml`: Code formatting rules

## Code Coverage

I maintain test coverage across the codebase. The coverage report helps me ensure:
- All critical functionality is tested
- Edge cases are handled properly
- Code quality is maintained

You can view the detailed coverage report on [Codecov](https://codecov.io/gh/keltoumboukra/protein-optimizer).

## Future Roadmap

I'm planning to integrate with key bioinformatics resources:
- AlphaFold for structure prediction
- UniProt for protein properties
- PDB for structural data
- ESM-2 for sequence analysis
- BRENDA for enzyme data
- KEGG for pathway information

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
