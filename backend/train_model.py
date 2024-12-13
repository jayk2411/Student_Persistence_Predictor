import os
import sys
import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def setup_logging():
    """Set up logging configuration"""
    if not os.path.exists('../logs'):
        os.makedirs('../logs')
    
    # Set up logging configuration
    log_filename = f'../logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

def clean_data(data_path):
    """Load and clean the data."""
    try:
        # Skip the first 24 rows which contain metadata
        df = pd.read_csv(data_path, skiprows=24, header=None)
        
        # Assign column names (matching the CSV file's structure)
        columns = [
            'First Term Gpa', 
            'Second Term Gpa', 
            'First Language', 
            'Funding',
            'School', 
            'FastTrack', 
            'Coop', 
            'Residency', 
            'Gender',
            'Previous Education', 
            'Age Group',
            'High School Average Mark',
            'Math Score',
            'English Grade',
            'FirstYearPersistence'
        ]
        
        df.columns = columns
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with median for each column
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        
        # Log data information
        logging.info(f"Data shape after cleaning: {df.shape}")
        logging.info("\nMissing values after cleaning:")
        logging.info(df.isnull().sum())
        logging.info("\nData statistics:")
        logging.info(df.describe())
        
        return df
        
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise

def prepare_data(df):
    """Prepare data for training."""
    try:
        # Select features and target
        features = [
            'First Term Gpa',
            'Second Term Gpa',
            'High School Average Mark',
            'Math Score'
        ]
        
        X = df[features]
        y = df['FirstYearPersistence']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        
    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise

def test_model(nn):
    """Test the trained model with sample cases."""
    logging.info("\nTesting model with sample cases...")
    
    # Test cases
    test_cases = [
        {
            'first_term_gpa': 3.5,
            'second_term_gpa': 3.2,
            'high_school_average': 85,
            'math_score': 40,
            'expected_result': 'high'
        },
        {
            'first_term_gpa': 1.5,
            'second_term_gpa': 1.2,
            'high_school_average': 60,
            'math_score': 20,
            'expected_result': 'low'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            # Prepare input data
            input_data = {k: v for k, v in case.items() if k != 'expected_result'}
            
            # Make prediction
            prediction = nn.predict(input_data)
            
            logging.info(f"\nTest Case {i}:")
            logging.info(f"Input: {input_data}")
            logging.info(f"Prediction: {prediction:.2%} probability of persistence")
            logging.info(f"Expected Range: {case['expected_result']}")
            
        except Exception as e:
            logging.error(f"Error testing case {i}: {str(e)}")

def verify_model_files():
    """Verify that model files were created and are accessible."""
    required_files = [
        'F:/STUDY/Sem-3/neural network/Project/StudentPersistencePredictor/models/persistence_model.h5',
        'F:/STUDY/Sem-3/neural network/Project/StudentPersistencePredictor/models/scaler.pkl'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"Missing required model file: {file_path}")
            return False
        else:
            file_size = os.path.getsize(file_path)
            logging.info(f"Found {os.path.basename(file_path)} (Size: {file_size/1024:.2f} KB)")
    
    return True

def train():
    """Main training function."""
    try:
        setup_logging()
        logging.info("Starting model training...")
        
        # Load and clean data
        data_path = 'F:/STUDY/Sem-3/neural network/Project/StudentPersistencePredictor/data/Student_data.csv'
        logging.info(f"Loading data from: {data_path}")
        
        df = clean_data(data_path)
        
        # Prepare data for training
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        
        # Initialize neural network
        nn = NeuralNetwork()
        
        # Train the model
        train_score = nn.train(X_train, y_train)
        
        # Evaluate the model
        test_loss, test_accuracy = nn.evaluate(X_test, y_test)
        
        # Set the scaler
        nn.set_scaler(scaler)
        
        logging.info(f"Training accuracy: {train_score:.4f}")
        logging.info(f"Test accuracy: {test_accuracy:.4f}")
        logging.info(f"Test loss: {test_loss:.4f}")
        
        # Save model and scaler
        nn.save_model(
            model_path='F:/STUDY/Sem-3/neural network/Project/StudentPersistencePredictor/models/persistence_model.h5',
            scaler_path='F:/STUDY/Sem-3/neural network/Project/StudentPersistencePredictor/models/scaler.pkl'
        )
        
        logging.info("Model and scaler saved successfully!")
        
        # Test the model
        test_model(nn)
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Train the model
        train()
        
        # Verify model files
        if verify_model_files():
            logging.info("Model training and verification completed successfully!")
        else:
            logging.error("Model training completed but some files are missing!")
    except Exception as e:
        logging.error(f"Training process failed: {str(e)}")
        sys.exit(1)