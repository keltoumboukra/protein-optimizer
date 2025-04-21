"""
Streamlit dashboard for protein expression optimization.

This module provides a web interface for:
- Visualizing protein expression data
- Making predictions for new protein expression conditions
- Analyzing feature importance
- Tracking prediction history
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import sys
import os
import json
from fastapi import FastAPI
from prediction_history import PredictionHistory

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize prediction history
prediction_history = PredictionHistory()

# Page config
st.set_page_config(
    page_title="Protein Expression Optimizer",
    page_icon="🧬",
    layout="wide",
)

# Title and description
st.title("Protein Expression Optimizer")
st.markdown(
    """
    This dashboard helps you optimize protein expression conditions using machine learning.
    Enter your experimental parameters below to get predictions for expression level and solubility.
    """
)

# Create tabs
tab1, tab2 = st.tabs(["Predictions", "Training Data"])

with tab1:
    # Load data from API
    @st.cache_data
    def load_data() -> pd.DataFrame:
        """
        Load protein expression data from the API.

        Returns:
            DataFrame containing protein expression samples
        """
        try:
            response = requests.get("http://localhost:8000/generate-sample")
            if response.status_code == 200:
                return pd.DataFrame([response.json()])
            else:
                st.error("Error loading data from API")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    # Load data
    df = load_data()

    if not df.empty:
        # Display sample data
        st.subheader("Sample Data")
        st.dataframe(df)

        # Create input form
        st.subheader("Enter Experimental Parameters")
        col1, col2 = st.columns(2)

        with col1:
            host_organism = st.selectbox(
                "Host Organism",
                options=df["host_organism"].unique(),
                index=0,
            )
            vector_type = st.selectbox(
                "Vector Type",
                options=df["vector_type"].unique(),
                index=0,
            )
            induction_condition = st.selectbox(
                "Induction Condition",
                options=df["induction_condition"].unique(),
                index=0,
            )

        with col2:
            media_type = st.selectbox(
                "Media Type",
                options=df["media_type"].unique(),
                index=0,
            )
            temperature = st.slider(
                "Temperature (°C)",
                min_value=20.0,
                max_value=37.0,
                value=37.0,
                step=0.1,
            )
            induction_time = st.slider(
                "Induction Time (hours)",
                min_value=1.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
            )

        # Make prediction
        if st.button("Predict"):
            try:
                # Prepare request data
                request_data = {
                    "host_organism": host_organism,
                    "vector_type": vector_type,
                    "induction_condition": induction_condition,
                    "media_type": media_type,
                    "temperature": temperature,
                    "induction_time": induction_time,
                }

                # Send request to API
                response = requests.post(
                    "http://localhost:8000/predict",
                    json=request_data,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted Expression Level",
                            f"{result['predicted_expression_level']:.1f}%",
                        )
                    with col2:
                        st.metric(
                            "Predicted Solubility",
                            f"{result['predicted_solubility']:.1f}%",
                        )

                    # Display feature importance
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(result["feature_importance"].items()),
                        columns=["Feature", "Importance"],
                    )
                    fig = px.bar(
                        importance_df,
                        x="Feature",
                        y="Importance",
                        title="Feature Importance Scores",
                    )
                    st.plotly_chart(fig)

                else:
                    st.error("Error making prediction")

            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    # Load experiment data
    @st.cache_data
    def load_experiments() -> pd.DataFrame:
        """
        Load experiment data from the API.

        Returns:
            DataFrame containing experiment metadata
        """
        try:
            response = requests.get("http://localhost:8000/experiments")
            if response.status_code == 200:
                data = response.json()
                return pd.DataFrame(data["experiments"])
            else:
                st.error("Error loading experiments from API")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading experiments: {str(e)}")
            return pd.DataFrame()

    # Load experiments
    experiments_df = load_experiments()

    if not experiments_df.empty:
        st.subheader("Training Data Sources")
        st.dataframe(experiments_df)

        # Display experiment details
        selected_exp = st.selectbox(
            "Select Experiment",
            options=experiments_df["id"],
            format_func=lambda x: f"{x} - {experiments_df[experiments_df['id'] == x]['name'].iloc[0]}",
        )

        if selected_exp:
            exp_data = experiments_df[experiments_df["id"] == selected_exp].iloc[0]
            st.markdown(f"**Description:** {exp_data['description']}")
            st.markdown(f"**Species:** {exp_data['species']}")
    else:
        st.error("No experiment data available")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Protein Expression Optimizer",
        description="Dashboard for protein expression optimization",
        version="1.0.0",
    )
    return app
