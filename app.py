# smart_agriculture_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("Data.csv")
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("🚜 Smart Agriculture App")
section = st.sidebar.radio("Select a Feature", [
    "📊 Data Overview",
    "🌾 Yield Predictor",
    "💧 Irrigation Forecast",
    "🧪 Pesticide Estimator",
    "💰 ROI Calculator"
])

# Encode categoricals
label_encoders = {}
categorical_cols = ['crop_type', 'irrigation_type', 'fertilizer_type', 'crop_disease_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop unnecessary columns
df = df.drop(columns=['farm_id', 'sensor_id', 'sowing_date', 'harvest_date', 'timestamp'])

# Shared default values
default_values = df.drop(columns=['yield_kg_per_hectare']).mean().to_dict()

# Section: Data Overview
if section == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

# Section: Yield Predictor
elif section == "🌾 Yield Predictor":
    st.header("🌾 Crop Yield Prediction")

    crop_type = st.selectbox("Select Crop Type", label_encoders['crop_type'].classes_)
    fert_type = st.selectbox("Select Fertilizer Type", label_encoders['fertilizer_type'].classes_)
    irri_type = st.selectbox("Select Irrigation Type", label_encoders['irrigation_type'].classes_)
    future_date = st.date_input("Select Future Harvest Date", value=datetime.today())

    total_days = max(1, (future_date - datetime.today().date()).days)

    new_input = default_values.copy()
    new_input.update({
        'crop_type': label_encoders['crop_type'].transform([crop_type])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert_type])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri_type])[0],
        'total_days': total_days
    })

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df.drop(columns=['yield_kg_per_hectare'])
    y = df['yield_kg_per_hectare']
    model.fit(X, y)

    input_df = pd.DataFrame([new_input])
    predicted_yield = model.predict(input_df)[0]
    maunds_per_acre = (predicted_yield / 40) / 2.47105
    st.success(f"Predicted Crop Yield: {maunds_per_acre:.2f} maunds/acre")

# Section: Irrigation Forecast
elif section == "💧 Irrigation Forecast":
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    # === Load Dataset ===
    file_path = r"Data.csv"
    df = pd.read_csv(file_path)

    # === Assign irrigation intervals based on typical Pakistani farming ===
    irrigation_intervals = {
        'Wheat': 28,
        'Sugarcane': 25,
        'Maize': 22,
        'Cotton': 24,
        'Rice': 10
    }
    df['irrigation_interval_days'] = df['crop_type'].map(irrigation_intervals)
    df = df.dropna(subset=['irrigation_interval_days'])

    # === Encode crop type ===
    le_crop = LabelEncoder()
    df['crop_type_encoded'] = le_crop.fit_transform(df['crop_type'])

    # === Select features and target ===
    features = ['crop_type_encoded', 'soil_moisture_%', 'temperature_C', 'rainfall_mm']
    X = df[features]
    y = df['irrigation_interval_days']

    # === Train Model ===
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # === Streamlit App ===
    st.title("🌾 Smart Farming - Irrigation Interval Predictor")

    # User Inputs
    crop_input = st.selectbox("Select Crop Type:", le_crop.classes_)
    soil_moisture = st.slider("Current Soil Moisture (%)", min_value=0, max_value=100, value=30)
    temperature = st.slider("Current Temperature (°C)", min_value=-10, max_value=50, value=25)
    rainfall = st.slider("Recent Rainfall (mm)", min_value=0, max_value=200, value=10)

    # Predict Button
    if st.button("Predict Irrigation Interval"):
        crop_encoded = le_crop.transform([crop_input])[0]
        input_df = pd.DataFrame([{
            'crop_type_encoded': crop_encoded,
            'soil_moisture_%': soil_moisture,
            'temperature_C': temperature,
            'rainfall_mm': rainfall
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"💧 Predicted irrigation should be done every **{prediction:.1f} days** based on current conditions.")

# Section: Pesticide Estimator
elif section == "🧪 Pesticide Estimator":
    st.header("🧪 Pesticide Usage Estimator")

    crop = st.selectbox("Crop Type", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer Type", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation Type", label_encoders['irrigation_type'].classes_)
    days = st.number_input("Total Days of Crop", min_value=1, value=90)

    df_pest = pd.read_csv("Smart_Farming_Crop_Yield_2024.csv")
    df_pest['irrigation_type'] = df_pest['irrigation_type'].replace("None", "Drip")
    df_pest = df_pest[['crop_type', 'fertilizer_type', 'irrigation_type', 'total_days', 'pesticide_usage_ml']].copy()
    df_pest['pesticide_usage_ml'] = pd.to_numeric(df_pest['pesticide_usage_ml'], errors='coerce') * 20

    le_map = {}
    for col in ['crop_type', 'fertilizer_type', 'irrigation_type']:
        le = LabelEncoder()
        df_pest[col] = le.fit_transform(df_pest[col])
        le_map[col] = le

    X = df_pest.drop(columns=['pesticide_usage_ml'])
    y = df_pest['pesticide_usage_ml']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    input_df = pd.DataFrame([{
        'crop_type': le_map['crop_type'].transform([crop])[0],
        'fertilizer_type': le_map['fertilizer_type'].transform([fert])[0],
        'irrigation_type': le_map['irrigation_type'].transform([irri])[0],
        'total_days': days
    }])

    predicted = model.predict(input_df)[0]
    per_week = predicted / days * 7
    per_acre = predicted / 2.47

    st.success(f"Total Pesticide Needed: {predicted:.2f} ml")
    st.info(f"Weekly Estimate: {per_week:.2f} ml/week")
    st.info(f"Per Acre Estimate: {per_acre:.2f} ml/acre")

# Section: ROI Calculator
elif section == "💰 ROI Calculator":
    st.header("💰 Return on Investment (ROI)")

    crop = st.selectbox("Crop Type", label_encoders['crop_type'].classes_)
    fert = st.selectbox("Fertilizer Type", label_encoders['fertilizer_type'].classes_)
    irri = st.selectbox("Irrigation Type", label_encoders['irrigation_type'].classes_)
    harvest_date = st.date_input("Expected Harvest Date", value=datetime.today())
    total_days = max(1, (harvest_date - datetime.today().date()).days)

    X = df.drop(columns=['yield_kg_per_hectare'])
    y = df['yield_kg_per_hectare']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    input_vals = default_values.copy()
    input_vals.update({
        'crop_type': label_encoders['crop_type'].transform([crop])[0],
        'fertilizer_type': label_encoders['fertilizer_type'].transform([fert])[0],
        'irrigation_type': label_encoders['irrigation_type'].transform([irri])[0],
        'total_days': total_days
    })

    input_df = pd.DataFrame([input_vals])
    predicted_yield = model.predict(input_df)[0]
    maunds = (predicted_yield / 40) / 2.47105

    prices = {'Wheat': 2000, 'Rice': 3600, 'Cotton': 8500, 'Maize': 1800, 'Sugarcane': 500}
    crop_name = crop.capitalize()
    price = prices.get(crop_name, 0)
    revenue = maunds * price

    st.success(f"Yield: {maunds:.2f} maunds/acre")
    st.info(f"Revenue: PKR {revenue:,.0f} per acre")

    cost = st.number_input("Enter total cost (PKR per acre)", value=50000.0)
    invest = st.number_input("Enter total investment (PKR per acre)", value=60000.0)

    profit = revenue - cost
    roi = (profit / invest) * 100 if invest else 0

    st.write("### ROI Results")
    st.write(f"Net Profit: PKR {profit:,.0f}")
    st.write(f"ROI: {roi:.2f}%")

    area = st.number_input("Enter total area (acres)", value=1.0)
    st.write("### Scaled Estimates")
    st.write(f"Total Revenue: PKR {revenue * area:,.0f}")
    st.write(f"Total Cost: PKR {cost * area:,.0f}")
    st.write(f"Total Investment: PKR {invest * area:,.0f}")
    st.write(f"Total Net Profit: PKR {profit * area:,.0f}")
