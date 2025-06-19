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

# ================= Yield Predictor ===================
st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
st.subheader("🌾 AI Yield Predictor")

input_data = {}
for feature in numeric_cols:
    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=float(round(default_values[feature], 2)))

if st.button("Predict Yield"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Yield: **{prediction:.2f} kg/hectare**")

# ================= Irrigation Interval Predictor ===================
st.markdown("</div>")
st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
st.subheader("💧 Irrigation Interval Estimator")

irrigation_intervals = {
    'Wheat': 28,
    'Sugarcane': 25,
    'Maize': 22,
    'Cotton': 24,
    'Rice': 10
}

crop_options = list(irrigation_intervals.keys())
crop_choice = st.selectbox("Select Crop Type", crop_options)
moisture = st.slider("Soil Moisture (%)", 0, 100, 30)
temp = st.slider("Temperature (°C)", -10, 50, 25)
rain = st.slider("Rainfall (mm)", 0, 200, 10)

irrigation_df = pd.DataFrame({
    'crop_type': [crop_choice],
    'soil_moisture_%': [moisture],
    'temperature_C': [temp],
    'rainfall_mm': [rain]
})

if st.button("Estimate Irrigation Interval"):
    base_interval = irrigation_intervals[crop_choice]
    adj = base_interval - (moisture / 10) + (rain / 50) - (temp / 20)
    st.info(f"Recommended Irrigation Interval: **{max(3, round(adj,1))} days**")

# ================= ROI Calculator ===================
st.markdown("</div>")
st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
st.subheader("💰 ROI & Cost Saving Calculator")

cost_fert = st.number_input("Cost of Fertilizer per kg", value=300.0)
cost_pest = st.number_input("Cost of Pesticide per kg", value=500.0)
cost_irrig = st.number_input("Irrigation Cost per Event", value=1000.0)
market_price = st.number_input("Market Price of Yield per kg", value=100.0)

avg_yield = df['yield_kg_per_hectare'].mean()
total_fert = df['fertilizer_usage_kg'].mean()
total_pest = df['pesticide_usage_kg'].mean()
total_irrig = 5  # assuming 5 irrigation events

total_input_cost = total_fert * cost_fert + total_pest * cost_pest + total_irrig * cost_irrig
income = avg_yield * market_price
profit = income - total_input_cost

st.metric("📈 Net Profit per Hectare", f"Rs. {profit:,.0f}")
st.metric("📊 Cost Efficiency", f"{(income / total_input_cost):.2f}x")

# ================= ROI & Dashboard Section ===================
st.markdown("</div>")
st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
st.subheader("📊 ROI & Monitoring Dashboard")

# 1. Average Soil Moisture by Day
daily_soil = df.groupby('Date')['soil_moisture_%'].mean().reset_index()
st.pyplot(sns.barplot(data=daily_soil, x='Date', y='soil_moisture_%', color='skyblue').figure)

# 2. Average Crop Yield by Day
daily_yield = df.groupby('Date')['yield_kg_per_hectare'].mean().reset_index()
st.pyplot(sns.barplot(data=daily_yield, x='Date', y='yield_kg_per_hectare', color='seagreen').figure)

# 3. Fertilizer Usage Boxplot
st.pyplot(sns.boxplot(y=df['fertilizer_usage_kg'], color='orange').figure)

# 4. Total Pesticide Usage
total_pest = df['pesticide_usage_kg'].sum()
fig4, ax4 = plt.subplots()
ax4.bar(['Pesticide Usage'], [total_pest], color='brown')
ax4.set_title('Total Pesticide Used in Season')
st.pyplot(fig4)

# 5. Pest Infestation Risk Pie Chart
risk_bins = pd.cut(df['pest_infestation_risk_%'], bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
risk_counts = risk_bins.value_counts()
fig5, ax5 = plt.subplots()
risk_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'gold', 'red'], ax=ax5)
ax5.set_ylabel('')
ax5.set_title('Pest Infestation Risk Levels')
st.pyplot(fig5)

# 6. Rainfall by Day
daily_rain = df.groupby('Date')['rainfall_mm'].mean().reset_index()
st.pyplot(sns.barplot(data=daily_rain, x='Date', y='rainfall_mm', color='dodgerblue').figure)

# 7. Soil Health Index Categories

def soil_category(val):
    if val >= 70: return 'Good'
    elif val >= 40: return 'Moderate'
    else: return 'Poor'

if 'soil_health_index' in df.columns:
    df['Soil Health Level'] = df['soil_health_index'].apply(soil_category)
    health_counts = df['Soil Health Level'].value_counts()
    st.pyplot(sns.barplot(x=health_counts.index, y=health_counts.values, palette='Set2').figure)

# 8. Irrigation Efficiency Histogram
if 'irrigation_efficiency_%' in df.columns:
    st.pyplot(sns.histplot(df['irrigation_efficiency_%'], bins=10, color='cornflowerblue').figure)

# 9. System Uptime Boxplot
if 'system_uptime_%' in df.columns:
    st.pyplot(sns.boxplot(y=df['system_uptime_%'], color='lightcoral').figure)

# 10. Daily Avg Temperature
if 'soil_temperature_°c' in df.columns:
    daily_temp = df.groupby('Date')['soil_temperature_°c'].mean().reset_index()
    fig10 = px.line(daily_temp, x='Date', y='soil_temperature_°c', title='Average Soil Temperature per Day')
    st.plotly_chart(fig10)

st.markdown("</div>", unsafe_allow_html=True)
