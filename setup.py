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
        "pandas",
        "numpy",
        "scikit-learn",
        "fastapi",
        "streamlit",
        "uvicorn",
        "plotly",
        "requests",
        "python-multipart",  # for FastAPI file uploads
        "urllib3>=2.0.0",  # Add explicit urllib3 requirement
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "mypy",
            "pandas-stubs",
        ],
    },
    python_requires=">=3.8",
)
