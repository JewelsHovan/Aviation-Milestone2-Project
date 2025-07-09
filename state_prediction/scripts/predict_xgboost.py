#!/usr/bin/env python
"""
Script: predict_xgboost.py
--------------------------
Uses trained XGBoost models to predict future flight states.
"""

import os
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
import xgboost as xgb

# Add parent directory to path
import sys
import fickling

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, DATA
from train_xgboost import engineer_features  # Reuse feature engineering function

def load_xgboost_models(target='all'):
    """
    Load trained XGBoost models.
    
    Args:
        target: Which target model(s) to load ('all' or specific target name)
        
    Returns:
        Dictionary of models or a single model
    """
    models = {}
    
    # Define the XGBoost models directory
    xgboost_dir = PATHS.model_dir / 'xgboost'
    
    if target == 'all':
        # Load all target models
        for target_name in DATA.target_columns:
            model_path = xgboost_dir / f'{target_name}_model.bin'
            if model_path.exists():
                model = xgb.Booster()  
                model.load_model(str(model_path))
                models[target_name] = model
            else:
                print(f"[WARNING] Model for {target_name} not found at {model_path}")
        
        # Also load feature list
        feature_path = xgboost_dir / f'{DATA.target_columns[0]}_features.json'
        with open(feature_path, 'r') as f:
            feature_list = json.load(f)
        
        return models, feature_list
    else:
        # Load specific target model
        model_path = xgboost_dir / f'{target}_model.bin'
        if model_path.exists():
            model = xgb.Booster()
            model.load_model(str(model_path))
            
            # Load feature list
            feature_path = xgboost_dir / f'{target}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return model, feature_list
        else:
            raise FileNotFoundError(f"Model for {target} not found at {model_path}")

def predict_next_state(sequence, models, feature_list, scalers=None):
    """
    Predict the next state given a sequence of previous states.
    
    Args:
        sequence: Numpy array of shape (seq_len, n_features)
        models: Dictionary of XGBoost models for each target or single model
        feature_list: List of feature names expected by the model
        scalers: Optional tuple of (X_scaler, y_scaler) for normalization
        
    Returns:
        Predicted next state as numpy array
    """
    # Add batch dimension to sequence
    sequence_batch = np.expand_dims(sequence, axis=0)  # (1, seq_len, n_features)
    
    # Engineer features
    feature_names = DATA.feature_columns
    X_df = engineer_features(sequence_batch, feature_names)
    
    # Make sure all expected features are present
    for feat in feature_list:
        if feat not in X_df.columns:
            X_df[feat] = 0  # Add missing feature with default value
    
    # Keep only the features used during training
    X_df = X_df[feature_list]
    
    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(X_df)
    
    # Predict using models
    if isinstance(models, dict):
        # Multiple target models
        predictions = np.zeros((1, len(DATA.target_columns)))
        for i, target_name in enumerate(DATA.target_columns):
            if target_name in models:
                predictions[0, i] = models[target_name].predict(dmatrix)[0]
    else:
        # Single target model
        predictions = models.predict(dmatrix)
    
    # Inverse transform if scalers provided
    if scalers is not None:
        X_scaler, y_scaler = scalers
        predictions = y_scaler.inverse_transform(predictions)
    
    return predictions[0] if len(predictions.shape) > 1 else predictions

def predict_trajectory(initial_sequence, models, feature_list, num_steps=10, scalers=None):
    """
    Predict a trajectory by recursively predicting future states.
    
    Args:
        initial_sequence: Initial sequence data (seq_len, n_features)
        models: Dictionary of models or single model
        feature_list: List of features expected by the model
        num_steps: Number of future steps to predict
        scalers: Optional tuple of (X_scaler, y_scaler) for normalization
        
    Returns:
        Array of predicted states
    """
    seq_len, n_features = initial_sequence.shape
    predictions = np.zeros((num_steps, n_features))
    
    # Make a copy of the initial sequence to avoid modifying it
    current_seq = initial_sequence.copy()
    
    for i in range(num_steps):
        # Predict next state
        next_state = predict_next_state(current_seq, models, feature_list, scalers)
        predictions[i] = next_state
        
        # Update sequence for next iteration
        current_seq = np.vstack([current_seq[1:], next_state])
    
    return predictions

def load_scalers():
    """Load the scalers used during training"""
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)
    return X_scaler, y_scaler

def parse_args():
    parser = argparse.ArgumentParser(description='Predict flight states using XGBoost models')
    parser.add_argument('--target', type=str, default='all',
                       choices=['all', 'lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target to predict (default: all)')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of future steps to predict (default: 10)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input sequence data (numpy file)')
    parser.add_argument('--output', type=str, default='predictions.npy',
                       help='Path to save predictions (default: predictions.npy)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load models
    print(f"[INFO] Loading XGBoost models for target: {args.target}")
    models, feature_list = load_xgboost_models(args.target)
    
    # Load scalers
    scalers = load_scalers()
    
    # Load input sequence
    print(f"[INFO] Loading input sequence from: {args.input}")
    if args.input.endswith('.npy'):
        input_sequence = np.load(args.input)
    else:
        raise ValueError("Input file must be a .npy file")
    
    # Predict trajectory
    print(f"[INFO] Predicting {args.steps} future steps")
    predictions = predict_trajectory(
        input_sequence, models, feature_list, args.steps, scalers
    )
    
    # Save predictions
    print(f"[INFO] Saving predictions to: {args.output}")
    np.save(args.output, predictions)
    
    print("[INFO] Prediction completed")

if __name__ == "__main__":
    main() 
