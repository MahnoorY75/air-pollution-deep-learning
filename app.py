import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="Air Pollution Health Predictor",
    layout="wide"
)

# CUSTOM STYLE
st.markdown("""
    <style>
    .main {
        background-color: #f7f7fb;
    }

    h1 {
        color: #7851a9;
        text-align: center;
    }

    .stButton>button {
        background-color: #7851a9;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    .stNumberInput label {
        color: #333;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# TITLE
st.title("🌫️ Air Pollution → Health Impact Prediction System")

st.write("Enter pollution values below and get predicted hospital admissions with visual insights.")

# LOAD MODEL
# 
model = tf.keras.models.load_model("pollution_model.h5")

# INPUT SECTION
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", value=100.0)
    no2 = st.number_input("NO2", value=50.0)
    co = st.number_input("CO", value=1.0)

with col2:
    pm10 = st.number_input("PM10", value=150.0)
    so2 = st.number_input("SO2", value=10.0)
    ozone = st.number_input("Ozone", value=40.0)

with col3:
    aqi = st.number_input("AQI", value=200.0)
    temp = st.number_input("Temperature (optional)", value=25.0)
    humidity = st.number_input("Humidity (optional)", value=60.0)

# PREDICTION BUTTON
# ----------------------------
if st.button("Predict Impact"):

    # Input array (7 features used in model)
    input_data = np.array([[pm25, pm10, no2, so2, co, ozone, aqi]])
    input_data = input_data.reshape((1, 1, 7))

    prediction = model.predict(input_data)[0][0]

    # RESULT DISPLAY
    # ----------------------------
    st.markdown("## 📊 Prediction Result")
    st.success(f"Predicted Hospital Admissions: {prediction:.2f}")

    # SIMPLE BAR CHART
    # ----------------------------
    st.markdown("## 📈 Pollution Levels Visualization")

    labels = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'AQI']
    values = [pm25, pm10, no2, so2, co, ozone, aqi]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=45)
    ax.set_title("Input Pollution Levels")
    st.pyplot(fig)

    # PREDICTION GRAPH
    
    st.markdown("## 📉 Prediction Insight Graph")

    fig2, ax2 = plt.subplots()

    ax2.plot([0, 1, 2], [prediction*0.8, prediction, prediction*1.1], marker='o')
    ax2.set_title("Hospital Admission Trend Simulation")
    ax2.set_ylabel("Admissions")

    st.pyplot(fig2)

    # SUMMARY BOX
    if prediction > 50:
        st.error("⚠️ High risk: Severe pollution impact detected")
    elif prediction > 20:
        st.warning("⚠️ Moderate risk: Monitor air quality")
    else:
        st.success("✅ Low risk: Safe conditions")