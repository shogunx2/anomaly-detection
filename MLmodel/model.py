import os
import json
import pickle
from datetime import datetime

from metaflow import FlowSpec, Parameter, step
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping 
import geohash2

#define paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


'''
Following steps in this model flow:
    1. Start
    2. Load data
    3. Build fetures
    4. Train model
    5. Evaluate model
    6. Save model
    7. End
'''

#Parameters that can be passed when running the flow

class AutoencoderFlow(FlowSpec):
    data_path = Parameter("data-path",
                help="Path to the dataset",
                default="/app/data/events.csv")
    
    epochs = Parameter("epochs",
                help="Number of epochs for training the model",
                default=50)
    
    batch_size = Parameter("batch-size",    
                help="Batch size for training the model",
                default=32)
    
    validation_split = Parameter("validation-split",
                help="Fraction of the data to be used for validation",
                default=0.2)
    
    anomaly_percentile = Parameter("anomaly-percentile",
                help="Percentile threshold for detecting anomalies",
                default=95)
    
    embedding_dim = 20
    
    @step
    def start(self):
        """
        Start step of the flow.
        """
        print("Starting the model flow...")
        self.next(self.load_data)
        
    
    @step
    def load_data(self):
        """
        Load the dataset from the specified path.
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Use resource_id as username
        if 'resource_id' in df.columns:
            df['username'] = df['resource_id'].astype(str)
            df['resource_id'] = df['resource_id'].astype(str)
        else:
            df['username'] = 'UNKNOWN'

        # Parse geoip JSON and extract longitude and latitude
        if 'geoip' in df.columns:
            geoip_data = df['geoip'].apply(json.loads)
            df['latitude'] = geoip_data.apply(lambda x: x.get('latitude'))
            df['longitude'] = geoip_data.apply(lambda x: x.get('longitude'))
            # Compute geohash with precision 3
            df['geohash'] = [
                geohash2.encode(lat, lon, precision=3)
                if pd.notnull(lat) and pd.notnull(lon) else None
                for lat, lon in zip(df['latitude'], df['longitude'])
            ]
        else:
            df['geohash'] = None

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['is_working_hour'] = df['timestamp'].dt.hour.apply(lambda h: 'True' if 9 <= h <= 18 else 'False')
        df['is_weekday'] = df['timestamp'].dt.dayofweek.apply(lambda d: 'True' if 0 <= d <= 4 else 'False')

        if 'enterprise_id' in df.columns:
            df['enterprise_id'] = df['enterprise_id'].astype(str)

        # Standardize 'success' column to string 'True'/'False'
        if 'success' in df.columns:
            df['success'] = df['success'].apply(
                lambda v: 'True' if v is True or (isinstance(v, str) and v.lower() == 'true')
                else ('False' if v is False or (isinstance(v, str) and v.lower() == 'false') else 'UNKNOWN')
            )

        # Drop columns not needed
        drop_cols = [
            'client_ip', 'country_code', 'latitude', 'longitude', 'hour', 'day_of_week', 'working_hour',
            'useragent', 'geoip', 'resource_name', 'resource_type', 'event_type'
        ]
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        features = ['username', 'geohash', 'success', 'resource_id', 'is_working_hour', 'is_weekday']
        for col in features:
            if col in df.columns:
                df[col] = df[col].astype(str)
            else:
                df[col] = 'UNKNOWN'

        self.df = df[features].copy()

        print(f"Loaded data with shape: {self.df.shape}")
        print(f"Data columns: {self.df.columns.tolist()}")

        self.next(self.build_features)
        
        
    @step
    def build_features(self):
        """
        Prepare features for the autoencoder.
        """
        
        df = self.df.copy()
        print("Building features...")
        
        self.categorical_features = list(df.columns)
        self.label_encoders = {}
        
        for col in self.categorical_features:
            le = LabelEncoder()
            unique_values = df[col].astype(str).unique().tolist()
            
            if col == 'success':
                if 'True' not in unique_values:
                    unique_values.append('True')
                if 'False' not in unique_values:
                    unique_values.append('False')
            if 'UNKNOWN' not in unique_values:
                unique_values.append('UNKNOWN')
            
            le.fit(unique_values)
            df[col] = le.transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Label encoder for '{col}' includes UNKNOWN token at index {np.where(le.classes_ == 'UNKNOWN')[0][0]}")
        
        self.vocab_sizes = {col: len(self.label_encoders[col].classes_) for col in self.categorical_features}
        print(f"Vocabulary sizes: {self.vocab_sizes}")
            
        self.processed_features_df = df[self.categorical_features]
        self.X = self.processed_features_df.values.astype(np.int32)
        indices = np.arange(len(self.X))
        train_idx, val_idx = train_test_split(indices, test_size=self.validation_split, random_state=42)
        self.X_train = self.X[train_idx]
        self.X_val = self.X[val_idx]
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Validation split: {len(self.X_val)} samples")
         
        self.next(self.train_model)
        
        
    @step
    def train_model(self):
        """
        Train the autoencoder model.
        """
        
        print("Building and training the feedforward autoencoder model...")
        
        input_categoricals = []
        embeddings = []
        output_layers = []
        
        # For each categorcial feature create input, embedding, and output softmax
        for i,col in enumerate(self.categorical_features):
            input_cat = Input(shape=(1,), name=f'{col}_input', dtype='int32')
            input_categoricals.append(input_cat)
            emb = Embedding(input_dim=self.vocab_sizes[col], output_dim=self.embedding_dim,
                            name=f'{col}_embedding')(input_cat)
            embeddings.append(Flatten()(emb))
            
        concatenated_embeddings = concatenate(embeddings, name="concatenate_embeddings")
        
        encoder = Dense(int(concatenated_embeddings.shape[-1]*0.75),activation='relu',
                        name = "encoder_dense1")(concatenated_embeddings)
        encoder = Dense(int(concatenated_embeddings.shape[-1]*0.5),activation='relu',
                        name = "encoder_dense2")(encoder)
        
        for i, col in enumerate(self.categorical_features):
            output = Dense(self.vocab_sizes[col], activation='softmax', name=f'{col}_output')(encoder)
            output_layers.append(output)
            
        self.autoencoder = Model(inputs = input_categoricals,outputs=output_layers)
        self.autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        print(self.autoencoder.summary())
        
        X_train_inputs = [self.X_train[:,i].reshape(-1,1) for i in range(self.X_train.shape[1])]
        X_val_inputs = [self.X_val[:,i].reshape(-1,1) for i in range(self.X_val.shape[1])]
        y_train = [self.X_train[:,i] for i in range(self.X_train.shape[1])]
        y_val = [self.X_val[:,i] for i in range(self.X_val.shape[1])]
        
        # early_stopping = EarlyStopping(
        #     monitor = 'val_loss',
        #     patience = 5,
        #     restore_best_weights = True,
        #     verbose = 1
        # )
        
        self.history = self.autoencoder.fit(
            X_train_inputs,
            y_train,
            epochs = self.epochs,
            batch_size = self.batch_size,
            shuffle = True,
            validation_data = (X_val_inputs, y_val),
            #callbacks = [early_stopping]
        )
        
        self.next(self.evaluate_model)
        
        
    @step
    def evaluate_model(self):
        """
        Evaluate the model and determine the anomaly threshold.
        """
        print("Evaluating model and determining anomaly threshold...")
        
        # Make predictions on all data
        X_inputs = [self.X[:, i].reshape(-1,1) for i in range(self.X.shape[1])]
        preds = self.autoencoder.predict(X_inputs)
        
        if isinstance(preds, np.ndarray):
            preds = [preds]
            
        n_mismatches = np.zeros(len(self.X), dtype=int)
        for i, col in enumerate(self.categorical_features):
            pred_indices = np.argmax(preds[i], axis=1)
            n_mismatches += (pred_indices != self.X[:, i])
            
        print(f"Number of mismatches per sample(first 10): {n_mismatches[:10]}")
        
        self.anomaly_threshold = 3
        anomalies = n_mismatches > self.anomaly_threshold
        
        print(f"Number of potential anomalies found: {np.sum(anomalies)} out of {len(n_mismatches)}")
        
        # Store evaluation results
        self.evaluation = {
            'anomaly_threshold': int(self.anomaly_threshold),
            'num_anomalies': int(np.sum(anomalies)),
            'total_samples': len(n_mismatches),
            'anomaly_percentage': float(100 * np.sum(anomalies) / len(n_mismatches)),
            'mismatches_per_sample': n_mismatches[:20].tolist()
        }
        
        self.next(self.save_model)
        
        
    @step
    def save_model(self):
        """
        Save the trained model and evaluation results.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(MODELS_DIR, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Saving model to {os.path.abspath(model_dir)}")
        
        # Save Keras model with extra options to ensute compatibility
        model_path = os.path.join(model_dir, "autoencoder.h5")
        self.autoencoder.save(model_path, save_format='h5', include_optimizer=False)
        print(f"Model saved to {model_path}")
        
        # Save model optimizer and loss information separately
        model_config = {
            'optimizer': 'adam',
            'loss': 'sparse_categorical_crossentropy',
            'metrics': []
        }
        with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
            json.dump(model_config, f)
        
        # Save preprocessing objects
        preprocessing_objects = {
            'label_encoders': self.label_encoders,
            'vocab_sizes': self.vocab_sizes,
            'embedding_dim': self.embedding_dim,
            'anomaly_threshold': self.anomaly_threshold,
            'categorical_features': self.categorical_features,
        }
        with open(os.path.join(model_dir, 'preprocessing_objects.pkl'), 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        #Save evaluation results
        with open(os.path.join(model_dir, 'evaluation.json'), 'w') as f:
            json.dump(self.evaluation, f, indent=2)
            
        # Save model summary
        model_summary = []
        self.autoencoder.summary(print_fn=lambda x: model_summary.append(x))
        with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as f:
            f.write("\n".join(model_summary))
            
        print("Model and evaluation results saved successfully!")
        
        # Save the path to the saved model for future reference
        self.model_path = model_dir
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End step of the flow.
        """
        print("Autoencoder model flow completed successfully!")
        print(f"Model saved at: {self.model_path}")
        print(f"Evaluation results: {json.dumps(self.evaluation, indent=2)}")
        
        
if __name__ == "__main__":
    AutoencoderFlow()