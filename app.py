# smart_agriculture_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.express as px

# Set modern page config
st.set_page_config(page_title="Smart Agriculture Dashboard", layout="wide")

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
default_values = df[numeric_cols].mean().to_dict()
X = df[numeric_cols]
y = df['yield_kg_per_hectare']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Custom CSS for background and toolbar
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1585881903788-225978740745');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .toolbar {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        display: flex;
        justify-content: space-around;
        z-index: 1000;
    }
    .toolbar button {
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
    }
    .toolbar button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Toolbar Navigation
st.markdown(
    """
    <div class="toolbar">
        <button onclick="window.location.href='#dashboard'">📈 Dashboard</button>
        <button onclick="window.location.href='#yield-predictor'">🌾 Yield Predictor</button>
        <button onclick="window.location.href='#irrigation-forecast'">💧 Irrigation Forecast</button>
        <button onclick="window.location.href='#pesticide-estimator'">🧪 Pesticide Estimator</button>
        <button onclick="window.location.href='#roi-calculator'">💰 ROI Calculator</button>
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Dashboard
if st.session_state.get("section") == "Dashboard" or not st.session_state.get("section"):
    st.title("📊 Smart Agriculture Dashboard")
    st.subheader("Interactive Crop Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df, x="crop_type", title="Crop Type Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(df, x="crop_type", y="yield_kg_per_hectare", title="Yield by Crop Type")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.scatter(df, x="rainfall_mm", y="soil_moisture_%", color="crop_type", 
                          title="Rainfall vs. Soil Moisture")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.density_contour(df, x="temperature_C", y="humidity_%", title="Temperature vs. Humidity")
        st.plotly_chart(fig4, use_container_width=True)

# 2. Yield Predictor
elif st.session_state.get("section") == "Yield Predictor":
    st.title("🌾 Crop Yield Predictor")

    crop_type = st.selectbox("Crop Type", label_encoders['crop_type'].classes_)
    fert_type = st.selectbox("Fertilizer Type", label_encoders['fertilizer_type'].classes_)
    irri_type = st.selectbox("Irrigation Type", label_encoders['irrigation_type'].classes_)
    future_date = st.date_input("Expected Harvest Date", datetime.today())

    total_days = max(1, (future_date - datetime.today().date()).days)
    new_input = default_values.copy()
    new_input.update({
        'crop_type': label_encoders['crop_type'].transform([crop_type])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert_type])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri_type])[0],
        'total_days': total_days
    })

    prediction = model.predict(pd.DataFrame([new_input]))[0]
    maunds = (prediction / 40) / 2.47105
    st.success(f"Predicted Yield: {maunds:.2f} maunds/acre")

# 3. Irrigation Forecast
elif st.session_state.get("section") == "Irrigation Forecast":
    st.title("💧 Irrigation Forecasting")

    irrigation_intervals = {
        'Wheat': 28,
        'Sugarcane': 25,
        'Maize': 22,
        'Cotton': 24,
        'Rice': 10
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

    crop_in = st.selectbox("Crop Type", le_crop.classes_)
    sm = st.slider("Soil Moisture (%)", 0, 100, 35)
    temp = st.slider("Temperature (°C)", -10, 50, 30)
    rain = st.slider("Rainfall (mm)", 0, 200, 15)

    encoded_crop = le_crop.transform([crop_in])[0]
    X_input = pd.DataFrame([[encoded_crop, sm, temp, rain]], columns=X_irrig.columns)
    result = model_irrig.predict(X_input)[0]
    st.info(f"Recommended irrigation every {result:.1f} days.")

# 4. Pesticide Estimator
elif st.session_state.get("section") == "Pesticide Estimator":
    st.title("🧪 Pesticide Usage Estimator")
    pest_df = df[['crop_type', 'fertilizer_type', 'irrigation_type', 'total_days', 'pesticide_usage_ml']].copy()
    pest_df['pesticide_usage_ml'] = pd.to_numeric(pest_df['pesticide_usage_ml'], errors='coerce') * 20

    X_p = pest_df.drop(columns='pesticide_usage_ml')
    y_p = pest_df['pesticide_usage_ml']
    model_p = RandomForestRegressor(n_estimators=100, random_state=42)
    model_p.fit(X_p, y_p)

    crop = st.selectbox("Crop", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation", label_encoders['irrigation_type'].classes_)
   
