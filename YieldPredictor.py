import streamlit as st
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

