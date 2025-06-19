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

# Sidebar Navigation
st.sidebar.title("🌱 Smart AgriTech")
section = st.sidebar.radio("📂 Choose Module", [
    "📈 Dashboard",
    "🌾 Yield Predictor",
    "💧 Irrigation Forecast",
    "🧪 Pesticide Estimator",
    "💰 ROI Calculator"
])

# Shared model data
default_values = df.drop(columns=['yield_kg_per_hectare']).mean().to_dict()
X = df.drop(columns=['yield_kg_per_hectare'])
y = df['yield_kg_per_hectare']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 1. Dashboard
if section == "📈 Dashboard":
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
elif section == "🌾 Yield Predictor":
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
elif section == "💧 Irrigation Forecast":
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
elif section == "🧪 Pesticide Estimator":
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
    days = st.number_input("Total Crop Duration", min_value=1, value=90)

    input_df = pd.DataFrame([{
        'crop_type': label_encoders['crop_type'].transform([crop])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
        'total_days': days
    }])

    pred_ml = model_p.predict(input_df)[0]
    st.success(f"Estimated pesticide required: {pred_ml:.2f} ml")

# 5. ROI Calculator
elif section == "💰 ROI Calculator":
    st.title("💰 ROI & Profit Estimator")

    crop = st.selectbox("Crop", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation", label_encoders['irrigation_type'].classes_)
    end_date = st.date_input("Expected Harvest Date", datetime.today())

    days = max(1, (end_date - datetime.today().date()).days)
    input_vals = default_values.copy()
    input_vals.update({
        'crop_type': label_encoders['crop_type'].transform([crop])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
        'total_days': days
    })

    pred_yield = model.predict(pd.DataFrame([input_vals]))[0]
    maunds = (pred_yield / 40) / 2.47105

    price_map = {'Wheat': 2000, 'Rice': 3600, 'Cotton': 8500, 'Maize': 1800, 'Sugarcane': 500}
    price = price_map.get(crop.capitalize(), 0)
    revenue = maunds * price

    cost = st.number_input("Total Cost (PKR/acre)", value=50000.0)
    invest = st.number_input("Investment (PKR/acre)", value=60000.0)

    profit = revenue - cost
    roi = (profit / invest) * 100 if invest else 0

    st.metric("Yield (maunds/acre)", f"{maunds:.2f}")
    st.metric("Revenue (PKR/acre)", f"{revenue:,.0f}")
    st.metric("Profit", f"{profit:,.0f}")
    st.metric("ROI", f"{roi:.2f}%")

    acres = st.slider("Scale (Acres)", 1, 100, 5)
    st.write("---")
    st.write(f"**Total Revenue:** PKR {revenue * acres:,.0f}")
    st.write(f"**Total Cost:** PKR {cost * acres:,.0f}")
    st.write(f"**Total Investment:** PKR {invest * acres:,.0f}")
    st.write(f"**Total Profit:** PKR {profit * acres:,.0f}")
