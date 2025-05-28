from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load('rf_spam_detector.joblib')

@app.route('/')
def home():
    return "Spam Detection API is running. Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data['comment']
    prediction = model.predict([comment])
    return jsonify({'spam': bool(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
