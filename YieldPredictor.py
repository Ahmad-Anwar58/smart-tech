# smart_agriculture_app.py

import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set modern page config
st.set_page_config(page_title="CropIQ – Intelligent Crop Yield Optimizer", layout="wide")

# Load and convert background image to Base64
@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_path = "background.jpg"
img_base64 = get_base64_image(img_path)

# Custom CSS with base64 background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }}
    .custom-section {{
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
    }}
    .center-title {{
        text-align: center;
        font-size: 2.5rem;
        color: #1e5631;
        font-weight: bold;
        margin-top: 2rem;
    }}
    .subheading-text {{
        font-size: 1.2rem;
        color: #222;
        text-align: center;
        margin-bottom: 2rem;
    }}
    .nav-bar {{
        background-color: #1e5631;
        color: white;
        padding: 1rem;
        font-size: 1.1rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
        border-radius: 0.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Data.csv")
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Date'] = df['timestamp'].dt.date
    return df

df = load_data()

# Encode categoricals
label_encoders = {}
categorical_cols = ['crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Remove non-numeric columns for default values and model training
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'yield_kg_per_hectare']

# Shared model data
default_values = df[numeric_cols].mean(numeric_only=True).to_dict()
X = df[numeric_cols]
y = df['yield_kg_per_hectare']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ================= Welcome Page ===================
st.markdown("""
<div class="nav-bar">
  <a style='text-decoration: none; color: white;'>🌾 Yield Predictor</a>
  <a style='text-decoration: none; color: white;'>💧 Irrigation Estimator</a>
  <a style='text-decoration: none; color: white;'>🧪 Pesticide Forecaster</a>
  <a style='text-decoration: none; color: white;'>📈 ROI & Dashboard</a>
</div>
<div class="center-title">🌾 CropIQ – Intelligent Crop Yield Optimizer</div>
<div class="subheading-text">
  A smart analytics and AI-powered platform that helps Pakistani farmers:
  <ul>
    <li>Predict crop yield</li>
    <li>Optimize irrigation, fertilizer, pesticide</li>
    <li>Get clear cost-saving recommendations</li>
    <li>Cluster plots by performance</li>
    <li>Receive personalized insights & PDF reports</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# === Import predictors and ROI modules ===
# Yield Predictor
exec(open("yield_predictor.py").read())

# Irrigation Forecast
exec(open("irrigation_forecast.py").read())

# Pesticide Estimator
exec(open("pesticide_estimator.py").read())

# ROI Calculator
exec(open("roi_calculator.py").read())

# Dashboard
exec(open("dashboard_charts.py").read())
