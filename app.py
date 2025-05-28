from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load('rf_spam_detector.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'comment' not in data:
        return jsonify({'error': 'No comment provided'}), 400
    
    comment = data['comment']
    try:
        prediction = model.predict([comment])
        return jsonify({'spam': bool(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
