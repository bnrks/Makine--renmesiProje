import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config 
st.set_page_config(page_title="Kalori Yakımı Tahmin Uygulaması", layout="wide")

# Load model
model = joblib.load('eniyi.joblib')

# Map exercise types to numeric values (same as training)
egzersiz_map = {
    "Kosu": 0,
    "Yuruyus": 1,
    "Yuzme": 2,
    "Bisiklet": 3,
    "Yoga": 4
}

def make_prediction(yas, kilo, egzersiz_turu, sure):
    # Create one-hot encoded features for exercise type
    features = np.zeros(8)  # Total features: 3 numeric + 5 exercise types
    features[0] = yas
    features[1] = kilo
    features[2] = sure
    features[3 + egzersiz_map[egzersiz_turu]] = 1
    return model.predict(features.reshape(1, -1))[0]

# UI Elements
st.title("Kalori Yakımı Tahmin Uygulaması")

# Sidebar inputs
st.sidebar.header("Tahmin Parametreleri")

yas = st.sidebar.number_input("Yaş:", min_value=18, max_value=70, value=30)
kilo = st.sidebar.number_input("Kilo (kg):", min_value=40.0, max_value=150.0, value=70.0, step=0.1)
egzersiz = st.sidebar.selectbox("Egzersiz Türü:", options=list(egzersiz_map.keys()))
sure = st.sidebar.number_input("Süre (dakika):", min_value=10, max_value=180, value=30)

if st.sidebar.button("Tahmin Et"):
    prediction = make_prediction(yas, kilo, egzersiz, sure)
    st.success(f"Tahmini Yakılan Kalori: {prediction:.2f} kcal")
    
    st.info(f"""
    Girilen Parametreler:
    - Yaş: {yas}
    - Kilo: {kilo:.1f} kg
    - Egzersiz: {egzersiz}
    - Süre: {sure} dakika
    """)