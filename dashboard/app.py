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

# Set page config FIRST!
st.set_page_config(
    page_title="Protein PTM Prediction Dashboard",
    page_icon="🧬",
    layout="wide",
)

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize prediction history
prediction_history = PredictionHistory()

# --- Load Data ---
@st.cache_data
def load_proteins_of_interest():
    tsv_path = os.path.join(os.path.dirname(__file__), '../assets/proteins_of_interest.tsv')
    return pd.read_csv(tsv_path, sep='\t')

@st.cache_data
def load_protein_details():
    json_path = os.path.join(os.path.dirname(__file__), '../assets/protein_details.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Only include entries with a non-empty 'accession' key
        return {entry['accession']: entry for entry in data if 'accession' in entry and entry['accession']}

proteins_df = load_proteins_of_interest()
protein_details = load_protein_details()

# --- Page config ---
st.title("🧬 Protein PTM Prediction Dashboard")

# --- Sidebar: Protein Search ---
st.sidebar.header("Protein Search")
accession = st.sidebar.selectbox(
    "Select a UniProt Accession:",
    proteins_df["UniProtKB Accession"].tolist(),
    format_func=lambda x: f"{x} - {proteins_df.loc[proteins_df['UniProtKB Accession'] == x, 'Protein Name'].values[0]}"
)

# --- Main: Protein Info and PTM Data ---
if accession:
    info = proteins_df[proteins_df["UniProtKB Accession"] == accession].iloc[0]
    st.subheader(f"Protein: {info['Protein Name']} ({accession})")
    st.write(f"Organism: {info['Organism']}")

    details = protein_details.get(accession)
    if details:
        # Show sequence length
        seq = details.get('sequence', {}).get('sequence', '')
        st.write(f"Sequence length: {len(seq)}")
        # Show PTM annotations
        ptms = [f for f in details.get('features', []) if f.get('type', '').lower() == 'modified residue']
        st.write(f"Number of PTM sites: {len(ptms)}")
        if ptms:
            ptm_df = pd.DataFrame(ptms)
            st.dataframe(ptm_df[['description', 'begin', 'end']])
        else:
            st.info("No PTM annotations found for this protein.")
        # --- PTM Site Visualization ---
        if seq and ptms:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(seq)+1)),
                y=[0]*len(seq),
                mode='lines',
                line=dict(color='lightgray'),
                showlegend=False
            ))
            for ptm in ptms:
                pos = int(ptm.get('begin', 0))
                fig.add_trace(go.Scatter(
                    x=[pos], y=[0],
                    mode='markers',
                    marker=dict(color='red', size=10),
                    name=ptm.get('description', 'PTM')
                ))
            fig.update_layout(
                title='PTM Sites on Sequence',
                xaxis_title='Residue Position',
                yaxis=dict(visible=False),
                showlegend=True,
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No details found for this protein.")

    # --- PTM Prediction ---
    st.subheader("PTM Prediction")
    if st.button("Predict PTM Status"):
        # Dummy prediction logic (replace with real model/API call)
        import random
        ptm_prob = random.uniform(0.1, 0.99)
        st.success(f"Predicted probability of PTM: {ptm_prob:.2f}")
        st.progress(ptm_prob)

# --- Hide old expression analysis and prediction forms ---
# (Commented out for PTM focus)
# ... existing code ...

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
