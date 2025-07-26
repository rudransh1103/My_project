
import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('fraud_model.pkl')

# Title
st.title("💳 Fraud Transaction Detection")

# Sidebar for input
st.sidebar.header("Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=1.0)
hour = st.sidebar.slider("Hour (0-23)", 0, 23)
day = st.sidebar.slider("Day (1-31)", 1, 31)
weekday = st.sidebar.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)))

# Prediction button
if st.sidebar.button("Predict Fraud"):
    input_data = np.array([[amount, hour, day, weekday]])
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("⚠️ This transaction is FRAUDULENT!")
    else:
        st.success("✅ This transaction is LEGITIMATE.")
