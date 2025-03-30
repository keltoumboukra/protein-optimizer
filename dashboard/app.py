import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="BugBrain Dashboard",
    page_icon="🐛",
    layout="wide"
)

# Title
st.title("🐛 BugBrain Dashboard")

# Sidebar
st.sidebar.header("Settings")
num_records = st.sidebar.slider("Number of Records", 10, 1000, 100)

# Generate mock data
@st.cache_data
def load_data(num_records):
    response = requests.get(f"http://localhost:8000/generate-sample")
    if response.status_code == 200:
        return pd.DataFrame([response.json()])
    return pd.DataFrame()

# Load data
df = load_data(num_records)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bug Distribution by Instrument")
    fig_instrument = px.pie(
        df,
        names="instrument",
        title="Bug Distribution by Instrument"
    )
    st.plotly_chart(fig_instrument, use_container_width=True)

with col2:
    st.subheader("Bug Distribution by Problem Type")
    fig_problem = px.pie(
        df,
        names="problem_type",
        title="Bug Distribution by Problem Type"
    )
    st.plotly_chart(fig_problem, use_container_width=True)

# Bug Prediction Form
st.subheader("Predict Bug Resolution Time")
with st.form("bug_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        instrument = st.selectbox(
            "Instrument",
            ["Hamilton", "Tecan", "Beckman", "Agilent", "PerkinElmer"]
        )
        problem_type = st.selectbox(
            "Problem Type",
            ["Hardware", "Software", "Calibration", "Sample Processing", "Communication"]
        )
    
    with col2:
        severity = st.selectbox(
            "Severity",
            ["Low", "Medium", "High", "Critical"]
        )
        status = st.selectbox(
            "Status",
            ["Open", "In Progress", "Resolved", "Closed"]
        )
    
    description = st.text_area("Description (Optional)")
    
    submitted = st.form_submit_button("Predict Resolution Time")
    
    if submitted:
        # Prepare request data
        bug_data = {
            "instrument": instrument,
            "problem_type": problem_type,
            "severity": severity,
            "status": status,
            "description": description
        }
        
        # Make API request
        response = requests.post(
            "http://localhost:8000/predict",
            json=bug_data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display prediction
            st.success(f"Predicted Resolution Time: {result['predicted_resolution_time']:.2f} hours")
            
            # Display feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame(
                list(result['feature_importance'].items()),
                columns=['Feature', 'Importance']
            )
            fig_importance = px.bar(
                importance_df,
                x='Feature',
                y='Importance',
                title="Feature Importance for Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.error("Error making prediction. Please try again.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for the Hackathon") 