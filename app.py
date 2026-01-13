import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Heart Predictor")

st.title("Heart Disease Prediction")
st.write("Enter patient details:")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
bp = st.slider("Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
hr = st.slider("Heart Rate", 60, 220, 150)

def load_model():
    try:
        with open("models/Random_Forest.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

if st.button("Predict"):
    model, scaler = load_model()
    if model and scaler:
        features = np.array([[age, 1 if sex=="Male" else 0, 0, bp, chol, 0, 0, hr, 0, 1.0, 1, 0, 2]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]
        if pred == 1:
            st.error(f"High Risk: {prob:.1%}")
        else:
            st.success(f"Low Risk: {prob:.1%}")
        st.progress(float(prob))
    else:
        st.error("Model not loaded")

st.write("---")
st.write("Final Project Submission.")

