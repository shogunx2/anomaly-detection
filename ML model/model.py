import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

class LoginAnomalyDetector(Model):
    # ...existing code...
    def __init__(self, input_dim):
        super(LoginAnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def predict_anomaly(self, data, threshold=0.05):
        reconstructions = self.predict(data)
        mse = tf.reduce_mean(tf.square(data - reconstructions), axis=1)
        return (mse > threshold).numpy(), mse.numpy()

if __name__ == "__main__":
    # ...training code...
    input_dim = 7
    model = LoginAnomalyDetector(input_dim)
    model.compile(optimizer='adam', loss='mae')

    # Sample training data (normal logins)
    X_train = np.random.normal(0, 1, (1000, input_dim))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model.fit(X_train_scaled, X_train_scaled, 
             epochs=20, 
             batch_size=32,
             validation_split=0.2)
    
    model.save('anomaly_model.keras')
    joblib.dump(scaler, 'scaler.save')