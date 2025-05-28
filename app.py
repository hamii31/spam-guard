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

# Create a form to accept input
with st.form(key='spam_form'):
    comment = st.text_area("Enter your comment:")
    submit_button = st.form_submit_button("Check for Spam")

    if submit_button:
        result = predict_spam(comment)
        st.json(result)
