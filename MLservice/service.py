import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from typing import Dict, Any
import geohash2
from flask import Flask, request, jsonify

#Utility: Feature engineering for a single event(dict)
def feature_engineering(event: Dict[str, Any],features: list) -> pd.DataFrame:
    row = {}

    if 'username' in features:
        # Use resource_id as username (matching model.py)
        row['username'] = str(event.get('resource_id','UNKNOWN'))

    if 'geohash' in features:
        geoip = event.get('geoip')
        if geoip:
            try:
                geoip_json = geoip if isinstance(geoip,dict) else json.loads(geoip)
                lat = geoip_json.get('latitude')
                lon = geoip_json.get('longitude')
                if lat and lon:
                    geohash = geohash2.encode(float(lat),float(lon),precision=3)
                    row['geohash'] = geohash
                else:
                    row['geohash'] = 'UNKNOWN'
            except Exception as e:
                row['geohash'] = 'UNKNOWN'
        else:
            row['geohash'] = 'UNKNOWN'

    if 'success' in features:
        val = event.get('success',None)
        if val is True or val is False:
            row['success'] = str(val)
        elif isinstance(val,str):
            if val.lower() == 'true':
                row['success'] = 'True'
            elif val.lower() == 'false':
                row['success'] = 'False'
            else:
                row['success'] = 'UNKNOWN'
        else:
            row['success'] = 'UNKNOWN'

    if 'resource_id' in features:
        # Use resource_id directly (matching model.py)
        row['resource_id'] = str(event.get('resource_id','UNKNOWN'))

    
    timestamp = event.get('timestamp')
    if 'is_working_hour' in features:
        try:
            ts = pd.to_datetime(timestamp)
            hour = ts.hour
            day_of_week = ts.dayofweek
            # Use same working hours as model.py (9-22)
            if 'is_working_hour' in features:
                row['is_working_hour'] = 'True' if 9 <= hour <= 22 else 'False'
            if 'is_weekday' in features:
                row['is_weekday'] = 'True' if day_of_week < 5 else 'False'
        except Exception as e:
            if 'is_working_hour' in features:
                row['is_working_hour'] = 'UNKNOWN'
            if 'is_weekday' in features:
                row['is_weekday'] = 'UNKNOWN'

    return pd.DataFrame([row])

class AutoencoderInference:
    def __init__(self,model_dir: str):
        self.model_dir = model_dir
        self.load_artifacts()
    
    def load_artifacts(self):
        model_path = os.path.join(self.model_dir,'autoencoder.h5')
        self.model = load_model(model_path)

        with open(os.path.join(self.model_dir, 'preprocessing_objects.pkl'), 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.label_encoders = preprocessors['label_encoders']
        self.vocab_sizes = preprocessors['vocab_sizes']
        self.embedding_dim = preprocessors['embedding_dim']
        self.anomaly_threshold = preprocessors.get('anomaly_threshold', None)
        self.categorical_features = preprocessors['categorical_features']

    def preprocess_event(self,event: Dict[str, Any]) -> np.ndarray:
        df = feature_engineering(event,self.categorical_features)
        for col in self.categorical_features:
            le = self.label_encoders[col]
            val = df.at[0,col]
            if val in le.classes_:
                df.at[0,col] = le.transform([val])[0]
            else:
                df.at[0,col] = le.transform(['UNKNOWN'])[0]

        X_inputs = [df[col].values.reshape(-1,1).astype(np.int32) for col in self.categorical_features]
        return X_inputs

    def predict_event(self,event: Dict[str, Any]) -> Dict[str, Any]:
        X_inputs = self.preprocess_event(event)
        original_event = {}

        for i, col in enumerate(self.categorical_features):
            original_event[col] = self.label_encoders[col].inverse_transform([int(X_inputs[i][0][0])])[0]

        preds = self.model.predict(X_inputs)
            
        if isinstance(preds, np.ndarray):
            preds = [preds]

        reconstructed_event = {}
        n_mismatches = 0
        mismatched_cols = []

        for i, col in enumerate(self.categorical_features):
            if i >= len(preds):
                print(f"Warning: No prediction for {col}(index {i})")
                pred_cat = 'UNKNOWN'
            elif preds[i].shape[0] == 0:
                pred_cat = 'UNKNOWN'
            else:
                try:
                    pred_idx = int(np.argmax(preds[i][0]))
                    pred_cat = self.label_encoders[col].inverse_transform([pred_idx])[0]
                except (IndexError, ValueError) as e:
                    print(f"Error processing prediction for {col}: {e}")
                    pred_cat = 'UNKNOWN'

            reconstructed_event[col] = pred_cat

            if pred_cat != original_event[col]:
                n_mismatches += 1
                mismatched_cols.append(col)

        # Use anomaly threshold from model if available, otherwise use 0 (since model shows 0 mismatches)
        threshold = self.anomaly_threshold if self.anomaly_threshold is not None else 0
        
        if n_mismatches == 1 and mismatched_cols == ['success']:
            is_anomaly = False
        else:
            is_anomaly = n_mismatches > threshold

        return {
            'original_event': original_event,
            'reconstructed_event': reconstructed_event,
            'is_anomaly': is_anomaly,
            'mismatched_cols': mismatched_cols,
            'n_mismatches': n_mismatches,
            'anomaly_threshold': threshold
        }

# Initialize Flask app
app = Flask(__name__)

# Initialize ML model
# Find the latest model directory
models_dir = "/app/models"
if os.path.exists(models_dir):
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith('model_')]
    if model_dirs:
        latest_model = sorted(model_dirs)[-1]
        model_path = os.path.join(models_dir, latest_model)
        print(f"Loading model from: {model_path}")
        inference = AutoencoderInference(model_path)
    else:
        print("No model directories found!")
        inference = None
else:
    print(f"Models directory not found: {models_dir}")
    inference = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict anomaly for an event"""
    if inference is None:
        return jsonify({
            'error': 'ML model not loaded',
            'is_anomaly': False
        }), 500
    
    try:
        event_data = request.json
        if not event_data:
            return jsonify({
                'error': 'No event data provided',
                'is_anomaly': False
            }), 400
        
        print(f"Received event for prediction: {event_data}")
        result = inference.predict_event(event_data)
        print(f"Prediction result: {result}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            'error': str(e),
            'is_anomaly': False
        }), 500

if __name__ == '__main__':
    print("Starting ML Service on port 8001...")
    app.run(host='0.0.0.0', port=8001, debug=False) 