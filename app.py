import streamlit as st
import joblib
import json

model = joblib.load('rf_spam_detector.joblib')

def predict_spam(comment):
    prediction = model.predict([comment])
    return {'spam': bool(prediction[0])}

# Detect query param access
query_params = st.experimental_get_query_params()

if "comment" in query_params:
    comment = query_params["comment"][0]
    result = predict_spam(comment)

    # Return raw JSON string for compatibility
    st.write("Content-type: application/json\n")
    st.json(result)

else:
    # Web UI for manual testing
    st.title("Spam Detection API")
    comment = st.text_area("Enter your comment:")
    if st.button("Check for Spam"):
        result = predict_spam(comment)
        st.json(result)
