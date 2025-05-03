import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
from datetime import datetime

# Load model and encoders
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

st.set_page_config(page_title="Financial Fraud Detector", page_icon=":money_with_wings:")
st.title("Financial Fraud Detector")
st.image("financial fraud detector logo.png", width=200)

# Input fields
st.header("Enter Transaction Details")

merchant = st.text_input("Merchant")
category = st.text_input("Category")
amt = st.number_input("Amount", min_value=0.0)
lat = st.number_input("User Latitude", format="%.6f")
long = st.number_input("User Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Hour of Day", 0, 23)
day = st.slider("Day of Month", 1, 31)
month = st.slider("Month", 1, 12)
gender = st.selectbox("Gender", ["M", "F"])
cc_num = st.text_input("Credit Card Number")

# Calculate distance
distance = geodesic((lat, long), (merch_lat, merch_long)).miles if all(v != 0.0 for v in [lat, long, merch_lat, merch_long]) else 0.0

# Predict button
if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])
        
        # Encode categorical columns
        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        # Hash credit card number
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Prediction
        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        st.subheader("Prediction Result:")
        st.success(f"The transaction is **{result}**.")
    else:
        st.error("Please fill all required fields.")


   
