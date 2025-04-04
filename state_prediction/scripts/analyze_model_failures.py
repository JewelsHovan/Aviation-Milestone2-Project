#!/usr/bin/env python
"""
Script: analyze_model_failures.py
---------------------------------
Analyzes failures in model predictions to understand what features
contribute most to prediction errors using SHAP values.

Works with all model types: transformer, lstm, ffnn, and xgboost.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
import shap
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path
import sys
import fickling

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import PATHS, DATA, MODEL, TRAIN

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model prediction failures')
    parser.add_argument('--model', type=str, required=True,
                       choices=['transformer', 'lstm', 'ffnn', 'xgboost'],
                       help='Model type to analyze')
    parser.add_argument('--target', type=str, default='all',
                       choices=['all', 'lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target variable to analyze (default: all)')
    parser.add_argument('--num_failures', type=int, default=20,
                       help='Number of worst predictions to analyze (default: 20)')
    parser.add_argument('--error_threshold', type=float, default=None,
                       help='Analyze predictions with errors above this threshold')
    parser.add_argument('--background_samples', type=int, default=100,
                       help='Number of background samples for SHAP (default: 100)')
    return parser.parse_args()

def load_data():
    """Load the test data and scalers"""
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)
        
    return X_test, y_test, X_scaler, y_scaler

def load_model(model_type, target=None):
    """Load the trained model"""
    if model_type == 'xgboost':
        if target == 'all':
            # Load all XGBoost models
            models = {}
            xgboost_dir = PATHS.model_dir / 'xgboost'
            for target_name in DATA.target_columns:
                model_path = xgboost_dir / f'{target_name}_model.bin'
                if model_path.exists():
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    models[target_name] = model
            
            # Load feature list
            feature_path = xgboost_dir / f'{DATA.target_columns[0]}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return models, feature_list
        else:
            # Load specific XGBoost model
            model_path = PATHS.model_dir / 'xgboost' / f'{target}_model.bin'
            if not model_path.exists():
                raise FileNotFoundError(f"Model for {target} not found at {model_path}")
            
            model = xgb.Booster()
            model.load_model(str(model_path))
            
            # Load feature list
            feature_path = PATHS.model_dir / 'xgboost' / f'{target}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return model, feature_list
    else:
        # Load neural network model
        # First load the model configuration
        config_path = PATHS.get_model_config_path(model_type)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update MODEL configuration with loaded values
        MODEL.model_type = model_type
        model_config = config[f"{model_type}_config"]
        
        # Set model parameters
        model_params = {
            "input_dim": MODEL.input_dim,
            **model_config
        }
        
        # Initialize model
        device = torch.device(TRAIN.device)
        model = get_model(model_type, **model_params).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(
            PATHS.get_model_weights_path(model_type),
            map_location=device
        ))
        
        model.eval()  # Set to evaluation mode
        return model, device

def get_predictions(model, X_test, model_type, device=None, target_idx=None):
    """Get model predictions"""
    if model_type == 'xgboost':
        if isinstance(model, dict):
            # Multiple XGBoost models
            predictions = np.zeros((X_test.shape[0], len(DATA.target_columns)))
            for i, target_name in enumerate(DATA.target_columns):
                if target_name in model:
                    # Engineer features for this model
                    from train_xgboost import engineer_features
                    X_test_df = engineer_features(X_test, DATA.feature_columns)
                    dtest = xgb.DMatrix(X_test_df)
                    predictions[:, i] = model[target_name].predict(dtest)
            return predictions
        else:
            # Single XGBoost model
            from train_xgboost import engineer_features
            X_test_df = engineer_features(X_test, DATA.feature_columns)
            dtest = xgb.DMatrix(X_test_df)
            predictions = model.predict(dtest)
            return predictions
    else:
        # Neural network model
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            predictions = model(X_tensor).cpu().numpy()
            
            if target_idx is not None:
                predictions = predictions[:, target_idx]
                
            return predictions

def analyze_failures_nn(model, X_test, y_test, model_type, target_name, target_idx, 
                       num_failures=20, error_threshold=None, background_samples=100):
    """Analyze failures for neural network models"""
    # Create directory for failure analysis
    failure_dir = PATHS.model_dir / model_type / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cpu')  # Use CPU for SHAP analysis
    model = model.to(device)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test.squeeze(1), dtype=torch.float32)
    
    # Get predictions in batches to avoid OOM errors
    batch_size = 64  # Adjust this based on your memory
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(device)
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)
    
    # Combine batch predictions
    predictions = np.vstack(predictions)
    
    # Get target predictions and actual values
    y_pred = predictions[:, target_idx]
    y_true = y_test.squeeze(1)[:, target_idx]
    
    # Calculate errors
    errors = np.abs(y_true - y_pred)
    
    # Get indices of failures
    if error_threshold is not None:
        failure_indices = np.where(errors > error_threshold)[0]
        if len(failure_indices) == 0:
            print(f"[WARNING] No predictions found with errors above {error_threshold}")
            return
    else:
        # Get top N worst predictions
        failure_indices = np.argsort(errors)[-num_failures:]
    
    print(f"[INFO] Analyzing {len(failure_indices)} failures")
    
    # Create a DataFrame with failures
    failure_df = pd.DataFrame({
        'Index': failure_indices,
        'True_Value': y_true[failure_indices],
        'Predicted_Value': y_pred[failure_indices],
        'Absolute_Error': errors[failure_indices]
    })
    failure_df.to_csv(failure_dir / 'worst_predictions.csv', index=False)
    
    # For PyTorch models, use GradientExplainer instead of DeepExplainer
    try:
        # Select a smaller subset for background
        background_indices = np.random.choice(len(X_test), min(background_samples, 50), replace=False)
        background = X_tensor[background_indices].to(device)
        
        # Create a PyTorch-compatible explainer
        explainer = shap.GradientExplainer(model, background)
        
        # Compute SHAP values for failures in batches
        all_shap_values = []
        batch_size = 10  # Smaller batch size for SHAP computation
        
        for i in range(0, len(failure_indices), batch_size):
            batch_indices = failure_indices[i:i+batch_size]
            batch_X = X_tensor[batch_indices].to(device)
            batch_shap = explainer.shap_values(batch_X)
            
            # For PyTorch models, shap_values might be a list of arrays or have unexpected shape
            if isinstance(batch_shap, list):
                # If it's a list, take the appropriate target index if possible
                if len(batch_shap) > target_idx:
                    batch_shap = batch_shap[target_idx]
                else:
                    batch_shap = batch_shap[0]  # Default to first output
            
            # Handle the case where SHAP returns shape (batch, seq_len, features, output_dim)
            if len(batch_shap.shape) == 4 and batch_shap.shape[3] == len(DATA.target_columns):
                # Extract the values for the specific target
                batch_shap = batch_shap[:, :, :, target_idx]
            
            # Ensure the batch_shap has the right shape
            expected_shape = batch_X.shape
            if batch_shap.shape != expected_shape:
                # Try to reshape if total elements match
                if np.prod(batch_shap.shape) == np.prod(expected_shape):
                    batch_shap = batch_shap.reshape(expected_shape)
                else:
                    print(f"[WARNING] Unexpected SHAP values shape: {batch_shap.shape}, expected: {expected_shape}")
                    # Skip this batch if shapes don't match
                    continue
            
            all_shap_values.append(batch_shap)
        
        # Combine all SHAP values
        if len(all_shap_values) > 0:
            shap_values = np.vstack(all_shap_values)
            
            # Create feature names for visualization
            feature_names = []
            for t in range(X_test.shape[1]):  # For each time step
                for f in range(X_test.shape[2]):  # For each feature
                    feature_name = f"{DATA.feature_columns[f]}_{t}"
                    feature_names.append(feature_name)
            
            # Create summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values.reshape(shap_values.shape[0], -1),
                X_test[failure_indices].reshape(len(failure_indices), -1),
                feature_names=feature_names,
                show=False
            )
            plt.title(f'SHAP Summary Plot for {target_name} Failures')
            plt.tight_layout()
            plt.savefig(failure_dir / 'shap_summary_plot.png')
            plt.close()
            
            # Analyze top contributing features
            shap_abs = np.abs(shap_values.reshape(shap_values.shape[0], -1))
            shap_mean = np.mean(shap_abs, axis=0)
            
            # Create DataFrame with feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': shap_mean
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            feature_importance.to_csv(failure_dir / 'feature_importance.csv', index=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['Feature'][:20][::-1], feature_importance['Importance'][:20][::-1])
            plt.title(f'Top 20 Features Contributing to {target_name} Prediction Errors')
            plt.tight_layout()
            plt.savefig(failure_dir / 'top_features.png')
            plt.close()
            
            print(f"[INFO] Top 5 features contributing to errors:")
            for feature, value in feature_importance.head(5).iterrows():
                print(f"  - {value['Feature']}: {value['Importance']:.6f}")
        
    except Exception as e:
        print(f"[WARNING] SHAP analysis failed: {str(e)}")
        print("[INFO] Falling back to simpler analysis method")
        
        # Fallback: Analyze input patterns for failures
        X_failures = X_test[failure_indices]
        
        # Calculate mean and std of each feature at each time step
        mean_features = np.mean(X_failures, axis=0)
        std_features = np.std(X_failures, axis=0)
        
        # Plot feature patterns for failures
        for f_idx, feature_name in enumerate(DATA.feature_columns):
            plt.figure(figsize=(10, 6))
            plt.plot(mean_features[:, f_idx], label='Mean')
            plt.fill_between(
                range(len(mean_features)),
                mean_features[:, f_idx] - std_features[:, f_idx],
                mean_features[:, f_idx] + std_features[:, f_idx],
                alpha=0.3
            )
            plt.title(f'Pattern of {feature_name} in Failed Predictions')
            plt.xlabel('Time Step')
            plt.ylabel(feature_name)
            plt.legend()
            plt.savefig(failure_dir / f'{feature_name}_pattern.png')
            plt.close()

def analyze_failures_xgb(model, X_test, y_test, target_name, target_idx, 
                       num_failures=20, error_threshold=None):
    """
    Analyze failures for XGBoost models using SHAP.
    
    Args:
        model: Trained XGBoost model
        X_test: Test input data
        y_test: Test target data
        target_name: Name of the target variable
        target_idx: Index of the target variable
        num_failures: Number of worst predictions to analyze
        error_threshold: Analyze predictions with errors above this threshold
    """
    from train_xgboost import engineer_features
    
    # Engineer features
    X_test_df = engineer_features(X_test, DATA.feature_columns)
    
    # Get predictions
    dtest = xgb.DMatrix(X_test_df)
    y_pred = model.predict(dtest)
    
    # Get actual values for the target
    y_actual = y_test.squeeze(1)[:, target_idx]
    
    # Calculate errors
    errors = np.abs(y_actual - y_pred)
    
    # Get indices of failures
    if error_threshold is not None:
        failure_indices = np.where(errors > error_threshold)[0]
        if len(failure_indices) == 0:
            print(f"[WARNING] No predictions found with errors above {error_threshold}")
            failure_indices = np.argsort(errors)[-num_failures:]
        elif len(failure_indices) > num_failures:
            # If too many failures above threshold, take the worst ones
            sorted_indices = failure_indices[np.argsort(errors[failure_indices])[::-1]]
            failure_indices = sorted_indices[:num_failures]
    else:
        # Get the worst num_failures predictions
        failure_indices = np.argsort(errors)[-num_failures:]
    
    # Create directory for failure analysis
    failure_dir = PATHS.model_dir / 'xgboost' / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary of worst predictions
    worst_predictions = pd.DataFrame({
        'True_Value': y_actual[failure_indices],
        'Predicted_Value': y_pred[failure_indices],
        'Absolute_Error': errors[failure_indices]
    })
    worst_predictions.to_csv(failure_dir / 'worst_predictions.csv', index=False)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for failure cases
    shap_values = explainer.shap_values(X_test_df.iloc[failure_indices])
    
    # Analyze each failure
    for i, idx in enumerate(range(len(failure_indices))):
        # Create force plot for this instance
        plt.figure(figsize=(14, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[idx], 
            X_test_df.iloc[failure_indices[idx]], 
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot for Prediction {i+1}\n'
                 f'True: {y_actual[failure_indices[idx]]:.4f}, '
                 f'Predicted: {y_pred[failure_indices[idx]]:.4f}, '
                 f'Error: {errors[failure_indices[idx]]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_force_plot.png')
        plt.close()
        
        # Create waterfall plot for this instance
        plt.figure(figsize=(14, 10))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_test_df.iloc[failure_indices[idx]].values,
                feature_names=X_test_df.columns
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for Prediction {i+1}\n'
                 f'True: {y_actual[failure_indices[idx]]:.4f}, '
                 f'Predicted: {y_pred[failure_indices[idx]]:.4f}, '
                 f'Error: {errors[failure_indices[idx]]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_waterfall_plot.png')
        plt.close()
    
    # Calculate feature contributions
    feature_contributions = pd.DataFrame(
        np.abs(shap_values), 
        columns=X_test_df.columns
    )
    
    # Calculate average absolute SHAP value for each feature
    avg_contributions = feature_contributions.mean().sort_values(ascending=False)
    
    # Plot average feature contributions to errors
    plt.figure(figsize=(12, 8))
    avg_contributions.head(20).plot(kind='bar')
    plt.title(f'Average Feature Contributions to Errors for {target_name}')
    plt.ylabel('Average |SHAP Value|')
    plt.tight_layout()
    plt.savefig(failure_dir / 'average_feature_contributions.png')
    plt.close()
    
    # Save feature contributions to CSV
    avg_contributions.to_csv(failure_dir / 'average_feature_contributions.csv')
    
    # Create summary plot for all failures
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_test_df.iloc[failure_indices],
        show=False
    )
    plt.title(f'SHAP Summary for {target_name} Failures')
    plt.tight_layout()
    plt.savefig(failure_dir / 'shap_summary.png')
    plt.close()
    
    print(f"[INFO] Analyzed {len(failure_indices)} failures for {target_name}")
    print(f"[INFO] Top 5 features contributing to errors:")
    for feature, value in avg_contributions.head(5).items():
        print(f"  - {feature}: {value:.6f}")

def main():
    # Parse command line arguments
    args = parse_args()
    model_type = args.model
    target = args.target
    
    # Load data
    print("[INFO] Loading test data...")
    X_test, y_test, X_scaler, y_scaler = load_data()
    
    # Load model
    print(f"[INFO] Loading {model_type} model...")
    model, extra_info = load_model(model_type, target)
    
    # Analyze failures
    if target == 'all':
        # Analyze all targets
        for i, target_name in enumerate(DATA.target_columns):
            print(f"[INFO] Analyzing failures for target: {target_name}")
            
            if model_type == 'xgboost':
                if target_name in model:
                    analyze_failures_xgb(
                        model[target_name], 
                        X_test, 
                        y_test, 
                        target_name, 
                        i,
                        num_failures=args.num_failures,
                        error_threshold=args.error_threshold
                    )
                else:
                    print(f"[WARNING] No model found for target: {target_name}")
            else:
                analyze_failures_nn(
                    model, 
                    X_test, 
                    y_test, 
                    model_type, 
                    target_name, 
                    i,
                    num_failures=args.num_failures,
                    error_threshold=args.error_threshold,
                    background_samples=args.background_samples
                )
    else:
        # Analyze specific target
        target_idx = DATA.target_columns.index(target)
        print(f"[INFO] Analyzing failures for target: {target}")
        
        if model_type == 'xgboost':
            analyze_failures_xgb(
                model, 
                X_test, 
                y_test, 
                target, 
                target_idx,
                num_failures=args.num_failures,
                error_threshold=args.error_threshold
            )
        else:
            analyze_failures_nn(
                model, 
                X_test, 
                y_test, 
                model_type, 
                target, 
                target_idx,
                num_failures=args.num_failures,
                error_threshold=args.error_threshold,
                background_samples=args.background_samples
            )
    
    print("[INFO] Failure analysis completed")

if __name__ == "__main__":
    main() 
