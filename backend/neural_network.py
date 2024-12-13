import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import os

class NeuralNetwork:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.history = None
        
    def build_model(self):
        """Build the neural network architecture"""
        self.model = keras.Sequential([
            keras.layers.Dense(8, activation='relu', input_shape=(4,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, X_train, y_train):
        """Train the neural network"""
        # Build the model if it doesn't exist
        if self.model is None:
            self.build_model()
        
         # Train the model and store history
        self.history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Return the history object
        return self.history
    
    def predict(self, input_data):
        """Make predictions using the trained model"""
        try:
            # Load model and scaler if not already loaded
            if self.model is None:
                self.load_model()
            
            # Prepare input features
            if isinstance(input_data, dict):
                features = np.array([[
                    float(input_data['first_term_gpa']),
                    float(input_data['second_term_gpa']),
                    float(input_data['high_school_average']),
                    float(input_data['math_score'])
                ]])
            else:
                features = input_data
                
            # Scale features
            if self.scaler:
                features = self.scaler.transform(features)
                
            # Make prediction
            prediction = float(self.model.predict(features)[0][0])
            return prediction
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def save_model(self, model_path='../models/persistence_model.h5', scaler_path='../models/scaler.pkl'):
        """Save the trained model and scaler"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and scaler
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path='../models/persistence_model.h5', scaler_path='../models/scaler.pkl'):
        """Load the trained model and scaler"""
        try:
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def set_scaler(self, scaler):
        """Set the scaler for feature normalization"""
        self.scaler = scaler
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise Exception("Model not trained or loaded")
            
        return self.model.evaluate(X_test, y_test, verbose=0)