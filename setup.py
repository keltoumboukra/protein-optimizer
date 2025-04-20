"""
Setup script for the protein-optimizer package.

This package provides tools for optimizing protein expression conditions using machine learning.
It includes a FastAPI backend for predictions and a Streamlit dashboard for visualization.
"""

from setuptools import setup, find_packages

setup(
    name="protein-optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Web Framework
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        
        # Data Processing and ML
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        
        # Dashboard
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            
            # Code Quality
            "black>=23.10.0",
            "mypy>=1.6.0",
            "pandas-stubs",
        ],
    },
    python_requires=">=3.9",
)
