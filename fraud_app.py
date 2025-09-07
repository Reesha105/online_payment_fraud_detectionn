# fraud_app.py
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("fraud_model.joblib")
features = joblib.load("features.pkl")

st.title("💳 Online Payment Fraud Detection")
st.write("Enter transaction details to check if it looks fraudulent.")

# Create inputs dynamically
user_input = {}
for f in features:
    user_input[f] = st.number_input(f"Enter {f}:", value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([user_input])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely Fraud (probability {proba:.2f})")
    else:
        st.success(f"✅ Legit Transaction (fraud probability {proba:.2f})")
