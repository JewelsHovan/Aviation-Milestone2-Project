#!/usr/bin/env python
"""
Script: forecast_single_flight.py
--------------------------------
Loads a single flight file, forecasts its trajectory starting from 50% 
of the flight duration, and compares the predicted path with the actual one.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import argparse
import json
import os
from pathlib import Path

# Add parent directory to path for model imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import PATHS, MODEL, TRAIN, INFERENCE

def parse_args():
    parser = argparse.ArgumentParser(description='Forecast trajectory for a single flight file')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    parser.add_argument('--flight_file', type=str, required=True, 
                       help='Path to the single flight file (parquet)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of steps to forecast')
    return parser.parse_args()

def load_config(model_path):
    """Load model configuration from JSON file"""
    if not model_path.exists():
        raise FileNotFoundError(f"No config file found at {model_path}")
    
    with open(model_path, 'r') as f:
        config = json.load(f)
    return config

def resample_flight_state_data(df, interval='3s'):
    """
    Resample flight state vector data to a specified time interval.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight state vector data.
        interval (str): Time interval for resampling.
        
    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    # Create a copy
    df_copy = df.copy()
    
    # Convert Unix timestamps to pandas datetime objects if needed
    if df_copy['time'].dtype != 'datetime64[ns]':
        df_copy['time'] = pd.to_datetime(df_copy['time'], unit='s')
    
    # Ensure data is sorted chronologically
    df_copy = df_copy.sort_values('time')
    
    # Set time as index for resampling
    df_copy.set_index('time', inplace=True)
    
    # Store original column order
    original_columns = df_copy.columns.tolist()
    
    # Resample data and forward fill missing values
    resampled_df = df_copy.resample(interval).first()
    
    # Reset index to make time a regular column again
    resampled_df.reset_index(inplace=True)
    
    # Ensure all original columns are present in the same order
    for col in original_columns:
        if col not in resampled_df.columns:
            resampled_df[col] = None
            
    resampled_df = resampled_df[['time'] + original_columns]
    
    return resampled_df

def preprocess_flight_data(df, feature_cols, input_len):
    """
    Preprocess flight data for model input.
    
    Args:
        df (pd.DataFrame): DataFrame containing flight data
        feature_cols (list): List of feature column names
        input_len (int): Length of input sequence
        
    Returns:
        np.array: Preprocessed sequence data
    """
    # Resample and interpolate
    df_res = resample_flight_state_data(df, interval='3s')
    
    # Set index
    df_res.set_index('time', inplace=True)
    
    # Interpolate columns
    for col in ['lat','lon','velocity','heading','vertrate','geoaltitude','baroaltitude']:
        if col in df_res.columns:
            df_res[col] = df_res[col].interpolate(method='time')
    
    # Drop rows with missing values
    df_res.dropna(subset=feature_cols, inplace=True)
    
    # Reset index
    df_res.reset_index(inplace=True)
    
    # Get feature values
    feature_values = df_res[feature_cols].values
    
    return df_res, feature_values

def generate_sequence(model, initial_sequence, y_scaler, num_predictions, device):
    """
    Generate autoregressive predictions.
    
    Args:
        model: Trained model
        initial_sequence: Initial sequence (1, seq_len, input_dim) scaled
        y_scaler: Scaler for the output
        num_predictions: Number of steps to predict
        device: Device to use
        
    Returns:
        np.array: Unscaled predictions
    """
    model.eval()
    if not isinstance(initial_sequence, torch.Tensor):
        initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
    initial_sequence = initial_sequence.to(device)

    predictions = []
    current_seq = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(num_predictions):
            next_pred = model(current_seq)  # (1, target_dim)
            predictions.append(next_pred.cpu().numpy())
            current_seq = current_seq.roll(-1, dims=1)
            current_seq[:, -1, :] = next_pred

    predictions = np.array(predictions).squeeze(axis=1)
    predictions_unscaled = y_scaler.inverse_transform(predictions)
    return predictions_unscaled

def plot_comparison(actual_path, predicted_path, start_point, save_path):
    """
    Plot actual vs predicted flight path.
    
    Args:
        actual_path: Actual flight path coordinates
        predicted_path: Predicted flight path coordinates
        start_point: Starting point for prediction
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot actual trajectory
    plt.plot(actual_path[:, 0], actual_path[:, 1], 'g-', linewidth=2, label='Actual Path')
    
    # Plot predicted trajectory
    full_pred = np.vstack([start_point, predicted_path])
    plt.plot(full_pred[:, 0], full_pred[:, 1], 'b--', linewidth=2, label='Predicted Path')
    
    # Mark start and end points
    plt.plot(start_point[0, 0], start_point[0, 1], 'go', markersize=10, label='Start Point')
    plt.plot(actual_path[-1, 0], actual_path[-1, 1], 'ro', markersize=8, label='Actual End')
    plt.plot(predicted_path[-1, 0], predicted_path[-1, 1], 'bo', markersize=8, label='Predicted End')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Trajectory Forecast Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved comparison plot to: {save_path}")
    
    # Also create a figure showing altitude over time
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(actual_path))
    pred_steps = np.arange(len(predicted_path)) + len(actual_path) // 2
    
    plt.plot(steps, actual_path[:, 5], 'g-', linewidth=2, label='Actual Altitude')
    plt.plot(pred_steps, predicted_path[:, 5], 'b--', linewidth=2, label='Predicted Altitude')
    
    plt.xlabel('Time Steps (3s intervals)')
    plt.ylabel('Geoaltitude')
    plt.title('Altitude Profile Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    altitude_plot_path = save_path.parent / f"{save_path.stem}_altitude{save_path.suffix}"
    plt.savefig(altitude_plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved altitude plot to: {altitude_plot_path}")

def calculate_metrics(actual, predicted):
    """
    Calculate error metrics between actual and predicted paths.
    
    Args:
        actual: Actual path values
        predicted: Predicted path values
        
    Returns:
        dict: Dictionary of error metrics
    """
    # Calculate Mean Squared Error
    mse = np.mean(np.sum((actual - predicted) ** 2, axis=1))
    
    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.sum(np.abs(actual - predicted), axis=1))
    
    # Calculate haversine distance for lat/lon (approximate)
    def haversine(lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    # Calculate average haversine distance
    avg_distance = np.mean([
        haversine(actual[i, 1], actual[i, 0], predicted[i, 1], predicted[i, 0])
        for i in range(len(actual))
    ])
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Avg_Distance_km": avg_distance
    }

def main():
    # Parse command line arguments
    args = parse_args()
    
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")
    
    # Load model configuration
    config = load_config(PATHS.get_model_config_path(args.model))
    
    # Verify model type matches
    if config["model_type"] != args.model:
        raise ValueError(f"Requested model type '{args.model}' doesn't match saved model type '{config['model_type']}'")
    
    # Update MODEL config with saved values
    MODEL.model_type = config["model_type"]
    MODEL.input_dim = config["input_dim"]
    model_specific_config = config[f"{MODEL.model_type}_config"]
    
    # 1. Load the single flight file
    flight_path = Path(args.flight_file)
    if not flight_path.exists():
        raise FileNotFoundError(f"Flight file not found: {flight_path}")
    
    print(f"[INFO] Loading flight file: {flight_path}")
    try:
        flight_df = pd.read_parquet(flight_path)
    except Exception as e:
        print(f"[ERROR] Failed to load flight file: {e}")
        sys.exit(1)
    
    # 2. Define feature columns
    feature_cols = ['lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude']
    input_len = 29  # Same as in prepare_data.py
    
    # 3. Preprocess the flight data
    print("[INFO] Preprocessing flight data...")
    df_processed, features = preprocess_flight_data(flight_df, feature_cols, input_len)
    
    # 4. Split at 50% of the flight
    total_steps = len(df_processed)
    midpoint = total_steps // 2
    
    print(f"[INFO] Flight has {total_steps} steps. Using first {midpoint} steps for input.")
    
    # 5. Load scalers
    print("[INFO] Loading scalers...")
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    # 6. Load the model
    print(f"[INFO] Loading {args.model} model...")
    if MODEL.model_type.lower() == "transformer":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.transformer)
        }
    elif MODEL.model_type.lower() == "lstm":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.lstm)
        }
    elif MODEL.model_type.lower() == "ffnn":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.ffnn)
        }
    elif MODEL.model_type.lower() == "kalman":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.kalman)
        }
    else:
        raise ValueError(f"Unknown model type: {MODEL.model_type}")
    
    model = get_model(MODEL.model_type, **model_params).to(device)
    model.load_state_dict(torch.load(
        PATHS.get_model_weights_path(args.model),
        map_location=device
    ))
    model.eval()
    
    # 7. Prepare input sequence (last input_len steps of first half)
    if midpoint < input_len:
        print(f"[WARNING] Flight too short! Need at least {input_len} steps.")
        sys.exit(1)
    
    # Use the last input_len steps from the first half of the flight
    input_sequence = features[midpoint - input_len:midpoint]
    
    # 8. Scale the input sequence
    input_sequence_scaled = X_scaler.transform(input_sequence)
    
    # 9. Reshape for model input
    input_tensor = np.expand_dims(input_sequence_scaled, axis=0)  # Add batch dimension
    
    # 10. Generate predictions
    print(f"[INFO] Generating {args.steps} prediction steps...")
    predictions = generate_sequence(
        model,
        input_tensor,
        y_scaler,
        num_predictions=args.steps,  # Use the specified steps or remaining flight length
        device=device
    )
    
    # 11. Get actual flight path after midpoint
    actual_path = features[midpoint:midpoint + args.steps] if midpoint + args.steps <= total_steps else features[midpoint:]
    
    # Trim predictions to match actual path length if needed
    if len(predictions) > len(actual_path):
        predictions = predictions[:len(actual_path)]
    
    # 12. Calculate metrics
    metrics = calculate_metrics(actual_path, predictions[:len(actual_path)])
    print("\n[INFO] Forecast Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # 13. Plot comparison
    print("[INFO] Plotting comparison...")
    last_input_point = input_sequence[-1:].reshape(1, -1)
    plot_comparison(
        actual_path,
        predictions,
        last_input_point,
        PATHS.model_dir / f"single_flight_forecast_{args.model}.png"
    )
    
    # 14. Save the results
    results = {
        "model_type": MODEL.model_type,
        "flight_file": str(flight_path),
        "total_steps": total_steps,
        "midpoint": midpoint,
        "prediction_steps": args.steps,
        "metrics": metrics
    }
    
    results_path = PATHS.model_dir / f"single_flight_forecast_{args.model}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[INFO] Saved forecast results to: {results_path}")

if __name__ == "__main__":
    main() 