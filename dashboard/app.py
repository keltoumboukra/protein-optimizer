"""
Streamlit dashboard for protein expression optimization.

This module provides a web interface for:
- Visualizing protein expression data
- Making predictions for new protein expression conditions
- Analyzing feature importance
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

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Protein Expression Optimization Dashboard",
    page_icon="🧬",
    layout="wide",
)

# Title
st.title("🧬 Protein Expression Optimization Dashboard")

# Sidebar
st.sidebar.header("Settings")
num_records = st.sidebar.slider("Number of Records", 10, 1000, 100)


# Generate mock data
@st.cache_data
def load_data(num_records: int) -> pd.DataFrame:
    """
    Load and cache protein expression data from the API.

    Args:
        num_records: Number of sample records to generate

    Returns:
        DataFrame containing protein expression samples
    """
    try:
        # Generate multiple samples to get enough data for visualization
        samples = []
        for _ in range(num_records):
            response = requests.get("http://localhost:8000/generate-sample")
            if response.status_code == 200:
                samples.append(response.json())

        if not samples:
            st.error("No data received from the API")
            return pd.DataFrame()

        return pd.DataFrame(samples)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


# Load data
df = load_data(num_records)

# Only show visualizations if we have data
if not df.empty:
    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Expression Distribution by Host Organism")
        try:
            fig_host = px.box(
                df,
                x="host_organism",
                y="expression_level",
                title="Expression Level Distribution by Host Organism",
            )
            st.plotly_chart(fig_host, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating host organism plot: {str(e)}")
            st.write("Available columns:", df.columns.tolist())

    with col2:
        st.subheader("Expression Distribution by Vector Type")
        try:
            fig_vector = px.box(
                df,
                x="vector_type",
                y="expression_level",
                title="Expression Level Distribution by Vector Type",
            )
            st.plotly_chart(fig_vector, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating vector type plot: {str(e)}")
            st.write("Available columns:", df.columns.tolist())

    # Expression Prediction Form
    st.subheader("Predict Protein Expression")
    with st.form("expression_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            host_organism = st.selectbox(
                "Host Organism",
                ["E. coli", "S. cerevisiae", "P. pastoris", "HEK293", "CHO"],
            )
            vector_type = st.selectbox(
                "Vector Type", ["pET", "pGEX", "pMAL", "pTrc", "pBAD"]
            )
            induction_condition = st.selectbox(
                "Induction Condition",
                ["IPTG", "Arabinose", "Methanol", "Galactose", "Tetracycline"],
            )

        with col2:
            media_type = st.selectbox("Media Type", ["LB", "TB", "M9", "YPD", "CD-CHO"])
            temperature = st.slider(
                "Temperature (°C)", min_value=20.0, max_value=37.0, value=37.0, step=0.5
            )
            induction_time = st.slider(
                "Induction Time (hours)",
                min_value=2.0,
                max_value=24.0,
                value=4.0,
                step=0.5,
            )

        description = st.text_area("Description (Optional)")

        submitted = st.form_submit_button("Predict Expression")

        if submitted:
            # Prepare request data
            experiment_data = {
                "host_organism": host_organism,
                "vector_type": vector_type,
                "induction_condition": induction_condition,
                "media_type": media_type,
                "temperature": temperature,
                "induction_time": induction_time,
                "description": description,
            }

            try:
                # Log the request data
                st.write(
                    "Sending request with data:", json.dumps(experiment_data, indent=2)
                )

                # Make API request
                response = requests.post(
                    "http://localhost:8000/predict", json=experiment_data
                )

                if response.status_code == 200:
                    result = response.json()

                    # Display predictions
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(
                            f"Predicted Expression Level: {result['predicted_expression_level']:.2f}%"
                        )
                    with col2:
                        st.success(
                            f"Predicted Solubility: {result['predicted_solubility']:.2f}%"
                        )

                    # Display feature importance
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(result["feature_importance"].items()),
                        columns=["Feature", "Importance"],
                    )
                    fig_importance = px.bar(
                        importance_df,
                        x="Feature",
                        y="Importance",
                        title="Feature Importance for Prediction",
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Error making prediction: {error_detail}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Full error details:", e)
else:
    st.warning(
        "Please make sure the FastAPI server is running on http://localhost:8000"
    )

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Protein Expression Optimizer",
        description="Dashboard for protein expression optimization",
        version="1.0.0"
    )
    return app
