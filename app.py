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
from pathlib import Path

# Set modern page config
st.set_page_config(page_title="Smart Agriculture Dashboard", layout="wide")

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

# Home Banner and Predictors
st.markdown("""
<div class="center-title">AGRICULTURE IS THE MOST HEALTHFUL</div>
""", unsafe_allow_html=True)

# Predictor Sections
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
        st.subheader("🌾 Yield Predictor")
        crop_type = st.selectbox("Crop Type", label_encoders['crop_type'].classes_, key="y_crop")
        fert_type = st.selectbox("Fertilizer Type", label_encoders['fertilizer_type'].classes_, key="y_fert")
        irri_type = st.selectbox("Irrigation Type", label_encoders['irrigation_type'].classes_, key="y_irri")
        future_date = st.date_input("Expected Harvest Date", datetime.today(), key="y_date")
        total_days = max(1, (future_date - datetime.today().date()).days)
        input_data = default_values.copy()
        input_data.update({
            'crop_type': label_encoders['crop_type'].transform([crop_type])[0],
            'fertilizer_type': label_encoders['fertilizer_type'].transform([fert_type])[0],
            'irrigation_type': label_encoders['irrigation_type'].transform([irri_type])[0],
            'total_days': total_days
        })
        pred = model.predict(pd.DataFrame([input_data]))[0]
        maunds = (pred / 40) / 2.47105
        st.success(f"Yield: {maunds:.2f} maunds/acre")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
        st.subheader("💧 Irrigation Forecast")
        irrigation_intervals = {
            'Wheat': 28, 'Sugarcane': 25, 'Maize': 22, 'Cotton': 24, 'Rice': 10
        }
        df_int = df.copy()
        crop_names = label_encoders['crop_type'].inverse_transform(df['crop_type'])
        df_int['crop_name'] = crop_names
        df_int['irrigation_interval_days'] = df_int['crop_name'].map(irrigation_intervals)
        df_int = df_int.dropna(subset=['irrigation_interval_days'])
        le_crop = LabelEncoder()
        df_int['crop_encoded'] = le_crop.fit_transform(df_int['crop_name'])
        X_irrig = df_int[['crop_encoded', 'soil_moisture_%', 'temperature_C', 'rainfall_mm']]
        y_irrig = df_int['irrigation_interval_days']
        model_irrig = RandomForestRegressor(n_estimators=100, random_state=42)
        model_irrig.fit(X_irrig, y_irrig)
        crop_in = st.selectbox("Crop", le_crop.classes_, key="i_crop")
        sm = st.slider("Soil Moisture (%)", 0, 100, 35, key="i_sm")
        temp = st.slider("Temperature (°C)", -10, 50, 30, key="i_temp")
        rain = st.slider("Rainfall (mm)", 0, 200, 15, key="i_rain")
        encoded_crop = le_crop.transform([crop_in])[0]
        X_input = pd.DataFrame([[encoded_crop, sm, temp, rain]], columns=X_irrig.columns)
        result = model_irrig.predict(X_input)[0]
        st.info(f"Irrigate every {result:.1f} days")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
        st.subheader("🧪 Pesticide Estimator")
        pest_df = df[['crop_type', 'fertilizer_type', 'irrigation_type', 'total_days', 'pesticide_usage_ml']].copy()
        pest_df['pesticide_usage_ml'] = pd.to_numeric(pest_df['pesticide_usage_ml'], errors='coerce') * 20
        X_p = pest_df.drop(columns='pesticide_usage_ml')
        y_p = pest_df['pesticide_usage_ml']
        model_p = RandomForestRegressor(n_estimators=100, random_state=42)
        model_p.fit(X_p, y_p)
        crop = st.selectbox("Crop", label_encoders['crop_type'].classes_, key="p_crop")
        fert = st.selectbox("Fertilizer", label_encoders['fertilizer_type'].classes_, key="p_fert")
        irri = st.selectbox("Irrigation", label_encoders['irrigation_type'].classes_, key="p_irri")
        days = st.number_input("Crop Duration (days)", min_value=1, value=90, key="p_days")
        input_df = pd.DataFrame([{
            'crop_type': label_encoders['crop_type'].transform([crop])[0],
            'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
            'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
            'total_days': days
        }])
        pred_ml = model_p.predict(input_df)[0]
        st.success(f"Estimated pesticide: {pred_ml:.2f} ml")
        st.markdown("</div>", unsafe_allow_html=True)

# Dashboard and ROI can be added similarly after this section
