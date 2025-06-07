import tensorflow as tf
import numpy as np
import joblib
from flask import Flask, request, jsonify
from model import LoginAnomalyDetector

app = Flask(__name__)
model = tf.keras.models.load_model('anomaly_model.keras', custom_objects={'LoginAnomalyDetector': LoginAnomalyDetector})
scaler = joblib.load('scaler.save')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'])
    data_scaled = scaler.transform(data)
    anomalies, scores = model.predict_anomaly(data_scaled)
    return jsonify({'anomalies': anomalies.tolist(), 'scores': scores.tolist()})

if __name__ == "__main__":
    app.run(debug=True)