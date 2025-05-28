# spam_api.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('rf_spam_detector.joblib')

@app.route('/predict', methods=['GET'])
def predict():
    comment = request.args.get('comment')
    if not comment:
        return jsonify({'error': 'No comment provided'}), 400
    prediction = model.predict([comment])
    return jsonify({'spam': bool(prediction[0])})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
