import joblib
import numpy as np

def load_scaler():
    """Load the saved StandardScaler"""
    return joblib.load("models/scaler.pkl")

def load_features():
    """Load the saved feature list"""
    return joblib.load("models/features.pkl")

def preprocess_input(input_dict):
    """
    input_dict: {feature_name: value} 
    Returns: scaled numpy array for PyTorch model
    """
    features = load_features()
    scaler = load_scaler()
    # Ensure correct order
    x = np.array([input_dict[f] for f in features]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return x_scaled
