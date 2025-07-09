#!/usr/bin/env python
"""
Script: train_xgboost.py
------------------------
Trains an XGBoost model for flight state prediction.

Unlike sequence-based models, this approach uses feature engineering
to transform sequential data into tabular format suitable for XGBoost.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import shap

# Add parent directory to path
import sys
import fickling

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, DATA

def parse_args():
    parser = argparse.ArgumentParser(description='Train XGBoost state prediction model')
    parser.add_argument('--target', type=str, default='all',
                       choices=['all', 'lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target variable to predict (default: all)')
    return parser.parse_args()

def load_raw_sequence_data():
    """Load the raw sequence data prepared for other models"""
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    # Load scalers for inverse transformation if needed
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)
        
    return X_train, y_train, X_test, y_test, X_scaler, y_scaler

def engineer_features(X_sequences, feature_names):
    """
    Transform sequence data into tabular format with engineered features.
    
    Args:
        X_sequences: Numpy array of shape (n_samples, seq_len, n_features)
        feature_names: List of feature names
        
    Returns:
        DataFrame with engineered features
    """
    n_samples, seq_len, n_features = X_sequences.shape
    feature_dfs = []
    
    for i in range(n_samples):
        seq = X_sequences[i]
        sample_features = {}
        
        # Last value for each feature
        for j, name in enumerate(feature_names):
            sample_features[f'{name}_last'] = seq[-1, j]
            
        # Statistical features for each variable
        for j, name in enumerate(feature_names):
            values = seq[:, j]
            sample_features[f'{name}_mean'] = np.mean(values)
            sample_features[f'{name}_std'] = np.std(values)
            sample_features[f'{name}_min'] = np.min(values)
            sample_features[f'{name}_max'] = np.max(values)
            
        # Rate of change features (delta between last few points)
        for j, name in enumerate(feature_names):
            values = seq[:, j]
            sample_features[f'{name}_delta1'] = values[-1] - values[-2]
            if seq_len > 2:
                sample_features[f'{name}_delta2'] = values[-1] - values[-3]
            if seq_len > 5:
                sample_features[f'{name}_delta5'] = values[-1] - values[-6]
                
        # Acceleration features (second derivative)
        for j, name in enumerate(feature_names):
            if seq_len > 2:
                delta1 = values[-1] - values[-2]
                delta2 = values[-2] - values[-3]
                sample_features[f'{name}_accel'] = delta1 - delta2
                
        # Add the sample features to our collection
        feature_dfs.append(pd.DataFrame([sample_features]))
    
    # Combine all samples into one dataframe
    return pd.concat(feature_dfs, ignore_index=True)

def train_xgboost_model(X_df, y, target_idx=None, valid_size=0.2):
    """
    Train an XGBoost model for a single target variable.
    
    Args:
        X_df: DataFrame with engineered features
        y: Target values (n_samples, n_targets)
        target_idx: Index of target to predict
        valid_size: Size of validation set as a fraction of training data
        
    Returns:
        Trained XGBoost model
    """
    # Extract the target variable
    if target_idx is not None:
        y_target = y[:, target_idx]
    else:
        # Default to first target if not specified
        y_target = y[:, 0]
    
    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_df, y_target, test_size=valid_size, random_state=42
    )
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    # Set up XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'seed': 42
    }
    
    # Train the model with validation data for early stopping
    evals = [(dtrain, 'train'), (dvalid, 'validation')]
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=500,
        early_stopping_rounds=20,
        evals=evals,
        verbose_eval=50  # Print progress every 50 iterations
    )
    
    return model

def evaluate_model(model, X_df, y_true, target_idx=None, target_name=None):
    """
    Evaluate the trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_df: DataFrame with engineered features
        y_true: True target values
        target_idx: Index of the target being predicted
        target_name: Name of the target variable
        
    Returns:
        Dictionary with evaluation metrics
    """
    dtest = xgb.DMatrix(X_df)
    y_pred = model.predict(dtest)
    
    if target_idx is not None:
        y_actual = y_true[:, target_idx]
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        
        # Make some basic plots
        plt.figure(figsize=(10, 6))
        plt.scatter(y_actual, y_pred, alpha=0.3)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
        plt.xlabel(f'True {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.title(f'XGBoost Prediction for {target_name} (RMSE={rmse:.4f})')
        plt.savefig(PATHS.model_dir / 'xgboost' / f'{target_name}_prediction.png')
        
        # SHAP analysis
        perform_shap_analysis(model, X_df, target_name)
        
        # Analyze worst predictions
        analyze_worst_predictions(X_df, y_actual, y_pred, model, target_name)
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae}
    else:
        # Multi-target evaluation
        metrics = {}
        for i, name in enumerate(DATA.target_columns):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            metrics[f'{name}_mse'] = mse
            metrics[f'{name}_rmse'] = np.sqrt(mse)
            metrics[f'{name}_mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        return metrics

def perform_shap_analysis(model, X_df, target_name):
    """
    Perform SHAP analysis on the model to understand feature importance.
    
    Args:
        model: Trained XGBoost model
        X_df: DataFrame with features
        target_name: Name of the target variable
    """
    print(f"[INFO] Performing SHAP analysis for {target_name}...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_df)
    
    # Create and save summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.title(f'SHAP Summary for {target_name}')
    plt.tight_layout()
    plt.savefig(PATHS.model_dir / 'xgboost' / f'{target_name}_shap_summary.png')
    plt.close()
    
    # Create and save feature importance plot based on SHAP
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance for {target_name}')
    plt.tight_layout()
    plt.savefig(PATHS.model_dir / 'xgboost' / f'{target_name}_shap_importance.png')
    plt.close()
    
    print(f"[INFO] SHAP analysis completed for {target_name}")

def analyze_worst_predictions(X_df, y_true, y_pred, model, target_name):
    """
    Analyze the worst predictions to understand failure modes.
    
    Args:
        X_df: DataFrame with features
        y_true: True target values
        y_pred: Predicted target values
        model: Trained XGBoost model
        target_name: Name of the target variable
    """
    print(f"[INFO] Analyzing worst predictions for {target_name}...")
    
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Get indices of worst predictions (top 20)
    worst_indices = np.argsort(errors)[-20:]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Create a directory for failure analysis
    failure_dir = PATHS.model_dir / 'xgboost' / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary of worst predictions
    worst_predictions = pd.DataFrame({
        'True_Value': y_true[worst_indices],
        'Predicted_Value': y_pred[worst_indices],
        'Absolute_Error': errors[worst_indices]
    })
    worst_predictions.to_csv(failure_dir / 'worst_predictions.csv', index=False)
    
    # Analyze each of the worst predictions
    for i, idx in enumerate(worst_indices):
        # Get SHAP values for this instance
        shap_values = explainer.shap_values(X_df.iloc[idx:idx+1])
        
        # Create force plot for this instance
        plt.figure(figsize=(14, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            X_df.iloc[idx], 
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot for Prediction {i+1}\n'
                 f'True: {y_true[idx]:.4f}, Predicted: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_force_plot.png')
        plt.close()
        
        # Create waterfall plot for this instance
        plt.figure(figsize=(14, 10))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_df.iloc[idx].values,
                feature_names=X_df.columns
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for Prediction {i+1}\n'
                 f'True: {y_true[idx]:.4f}, Predicted: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_waterfall_plot.png')
        plt.close()
    
    print(f"[INFO] Failure analysis completed for {target_name}")

def save_model_and_features(model, feature_list, target_name='all'):
    """Save the model and feature list"""
    # Create XGBoost model directory if it doesn't exist
    xgboost_dir = PATHS.model_dir / 'xgboost'
    xgboost_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = xgboost_dir / f'{target_name}_model.bin'
    model.save_model(str(model_path))
    
    # Save feature list
    with open(xgboost_dir / f'{target_name}_features.json', 'w') as f:
        json.dump(feature_list, f)
    
    print(f"[INFO] Saved XGBoost model and features for '{target_name}' target")

def main():
    # Parse command line arguments
    args = parse_args()
    target_var = args.target

    # Create model directory if it doesn't exist
    PATHS.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create XGBoost subdirectory
    xgboost_dir = PATHS.model_dir / 'xgboost'
    xgboost_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("[INFO] Loading sequence data...")
    X_train, y_train, X_test, y_test, X_scaler, y_scaler = load_raw_sequence_data()
    
    # Define feature names (same as target columns in this case)
    feature_names = DATA.feature_columns
    
    # 2. Engineer features
    print("[INFO] Engineering features from sequences...")
    X_train_df = engineer_features(X_train, feature_names)
    X_test_df = engineer_features(X_test, feature_names)
    
    # Get list of engineered features
    feature_list = X_train_df.columns.tolist()
    print(f"[INFO] Engineered {len(feature_list)} features")
    
    # 3. Train model based on target selection
    y_train_flat = y_train.squeeze(1)  # Remove seq_len dimension
    y_test_flat = y_test.squeeze(1)
    
    if target_var == 'all':
        # Train separate model for each target
        models = {}
        metrics = {}
        
        for i, target_name in enumerate(DATA.target_columns):
            print(f"[INFO] Training XGBoost model for target '{target_name}'...")
            model = train_xgboost_model(X_train_df, y_train_flat, i)
            models[target_name] = model
            
            # Evaluate each model
            dtest = xgb.DMatrix(X_test_df)
            y_pred = model.predict(dtest)
            mse = mean_squared_error(y_test_flat[:, i], y_pred)
            rmse = np.sqrt(mse)
            metrics[f'{target_name}_rmse'] = rmse
            
            # Save individual model
            save_model_and_features(model, feature_list, target_name)
            
            # Feature importance for this target
            plt.figure(figsize=(10, 12))
            xgb.plot_importance(model, max_num_features=20, height=0.8)
            plt.title(f'XGBoost Feature Importance ({target_name})')
            plt.tight_layout()
            plt.savefig(xgboost_dir / f'{target_name}_feature_importance.png')
        
        print("[INFO] Test metrics for all targets:", metrics)
    else:
        # Train model for specific target
        target_idx = DATA.target_columns.index(target_var)
        print(f"[INFO] Training XGBoost model for target '{target_var}'...")
        model = train_xgboost_model(X_train_df, y_train_flat, target_idx)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test_df, y_test_flat, target_idx, target_var)
        print(f"[INFO] Test RMSE for {target_var}: {metrics['rmse']:.4f}")
        
        # Save model
        save_model_and_features(model, feature_list, target_var)
        
        # Feature importance analysis
        plt.figure(figsize=(10, 12))
        xgb.plot_importance(model, max_num_features=20, height=0.8)
        plt.title(f'XGBoost Feature Importance ({target_var})')
        plt.tight_layout()
        plt.savefig(xgboost_dir / f'{target_var}_feature_importance.png')
    
    print("[INFO] Training completed")

if __name__ == "__main__":
    main() 
