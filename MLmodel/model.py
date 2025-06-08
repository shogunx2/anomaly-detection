import os
import json
import pickle
from datetime import datetime

from metaflow import FlowSpec, Parameter, step, current, IncludeFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, RepeatVector,TimeDistributed

#define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

class Model(FlowSpec):
    data_path = Parameter("data-path",
                help="Path to the dataset",
                default=os.path.join(ROOT_DIR, "data", "events.csv"))
    
    sequence_length = Parameter("sequence-length",
                help="Length of the input sequence for the model",
                default=10)
    
    epochs = Parameter("epochs",
                help="Number of epochs for training the model",
                default=10)
    
    batch_size = Parameter("batch-size",    
                help="Batch size for training the model",
                default=32)
    
    validation_split = Parameter("validation-split",
                help="Fraction of the data to be used for validation",
                default=0.2)
    
    anomaly_threshold = Parameter("anomaly-threshold",
                help="Threshold for detecting anomalies",
                default=0.95)
    
    embedding_dim = 64
    lstm_units = 64
    stride = 1
    
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
        
        #Flatten the 'GEOIP' and 'USERAGENT' columns
        
        try:
            geoip_flattened = pd.json_normalize(df['geoip'].apply(eval))
            useragent_flattened = pd.json_normalize(df['useragent'].apply(eval))
            
            df 
