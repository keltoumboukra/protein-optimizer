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
