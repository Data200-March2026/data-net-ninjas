import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ── Train model on startup ──────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('cleaned_dataset.csv')
    df['date_and_time'] = pd.to_datetime(df['date_and_time'])
    df['month'] = df['date_and_time'].dt.month
    df['season'] = df['month'].map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring', 4:'Spring', 5:'Spring',
        6:'Summer', 7:'Summer', 8:'Summer',
        9:'Autumn', 10:'Autumn', 11:'Autumn'
    })
    df['pm25_log'] = np.log(df['pm25'])
    features = ['temperature', 'wind_speed', 'relative_humidity',
                'precipitation', 'cloud_cover']
    X = df[features]
    y = df['pm25_log']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, df

model, df = load_and_train()

# ── AQI Category ────────────────────────────────────────────────────
def get_category(pm25):
    if pm25 <= 12:
        return "Good", "🟢", "#00e400"
    elif pm25 <= 35:
        return "Moderate", "🟡", "#ffff00"
    elif pm25 <= 55:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    else:
        return "Unhealthy", "🔴", "#ff0000"

# ── App Layout ───────────────────────────────────────────────────────
st.title("🌫️ Kathmandu Air Quality Predictor")
st.markdown("**DATA 200 Project — Net Ninjas**")
st.markdown("Enter current weather conditions to predict PM2.5 level")
st.divider()

# Input sliders
col1, col2 = st.columns(2)

with col1:
    temperature    = st.slider("🌡️ Temperature (°C)", 2, 30, 18)
    wind_speed     = st.slider("💨 Wind Speed (km/h)", 0, 15, 4)
    relative_humidity = st.slider("💧 Relative Humidity (%)", 24, 100, 78)

with col2:
    precipitation = st.slider("🌧️ Precipitation (mm)", 0, 30, 0)
    cloud_cover   = st.slider("☁️ Cloud Cover (%)", 0, 100, 40)

st.divider()

# Predict
if st.button("🔍 Predict Air Quality", use_container_width=True):
    input_data = pd.DataFrame([[temperature, wind_speed, relative_humidity,
                                 precipitation, cloud_cover]],
                               columns=['temperature', 'wind_speed',
                                        'relative_humidity', 'precipitation',
                                        'cloud_cover'])
    log_pred = model.predict(input_data)[0]
    pm25_pred = np.exp(log_pred)

    category, emoji, color = get_category(pm25_pred)

    st.markdown(f"""
    <div style='background-color:{color}33; padding:20px; border-radius:10px; text-align:center'>
        <h2>{emoji} {category}</h2>
        <h3>Predicted PM2.5: {pm25_pred:.1f} µg/m³</h3>
        <p>WHO Safe Limit: 15 µg/m³</p>
    </div>
    """, unsafe_allow_html=True)

    if pm25_pred > 15:
        st.warning("⚠️ PM2.5 is above WHO safe limit of 15 µg/m³")
    else:
        st.success("✅ PM2.5 is within WHO safe limit")

st.divider()
st.markdown("*Model: Linear Regression with Log Transformation | R² = 0.455*")