import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic
import base64
from PIL import Image

# Load model and encoders
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Display logo at the top of the app
def display_logo():
    file_path = "financial fraud detector logo.png"
    with open(file_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f'<div style="text-align: center; padding-bottom: 10px;"><img src="data:image/png;base64,{encoded}" width="150"/></div>',
            unsafe_allow_html=True
        )

# UI
st.set_page_config(page_title="Financial Fraud Detection System", layout="centered", page_icon="💳")
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        input, .stTextInput > div > div > input {
            background-color: #262730;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Render logo
display_logo()

st.markdown("<h1 style='text-align: center;'>Financial Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Enter the transaction details below</h5>", unsafe_allow_html=True)

# Input fields
merchant = st.text_input("Merchant Name")
category = st.selectbox("Category", ["Grocery", "Electronics", "Travel", "Food", "Clothing", "Other"])
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
with col2:
    long = st.number_input("Longitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")

col3, col4, col5 = st.columns(3)
with col3:
    hour = st.slider("Transaction Hour", 0, 23, 12)
with col4:
    day = st.slider("Transaction Day", 1, 31, 15)
with col5:
    month = st.slider("Transaction Month", 1, 12, 6)

gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card number")

# Predict button
distance = haversine(lat, long, merch_lat, merch_long)

if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "🔒 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.success(f"Prediction: {result}")
    else:
        st.error("Please fill all required fields.")

              

                
   
