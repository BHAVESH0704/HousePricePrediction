import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Simple ML app using Random Forest")

MODEL_PATH = "house_price_model.pkl"

# Check model existence
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.info("👉 Please run `python main.py` first to train and save the model.")
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)

st.success("✅ Model loaded successfully!")

# Inputs
area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 5, 2)
quality = st.slider("Overall Quality (1–10)", 1, 10, 7)

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame(
        [[area, bedrooms, bathrooms, quality]],
        columns=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']
    )

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹ {prediction:,.0f}/-")
