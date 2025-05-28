import streamlit as st
import joblib
import json

# Load your model
model = joblib.load('rf_spam_detector.joblib')

def predict_spam(comment):
    prediction = model.predict([comment])
    return {'spam': bool(prediction[0])}

# Streamlit app
st.title("Spam Detection API")

# Check if the request is from an API call
if st.button("Check for Spam"):
    comment = st.text_area("Enter your comment:")
    result = predict_spam(comment)
    st.json(result)

# For API requests
if st.experimental_get_query_params().get("comment"):
    comment = st.experimental_get_query_params()["comment"][0]
    result = predict_spam(comment)
    st.json(result)
