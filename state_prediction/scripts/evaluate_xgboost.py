#!/usr/bin/env python
"""
Script: evaluate_xgboost.py
---------------------------
Evaluates trained XGBoost models using k-fold cross validation and calculates
multiple error metrics including MSE, RMSE, and MAE.
"""

import os
import numpy as np
import pandas as pd
import json
import time
import argparse
import matplotlib
import fickling

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import xgboost as xgb

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, DATA
from train_xgboost import engineer_features

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: MSE, RMSE, MAE
    
    Args:
        y_true: Ground truth values (numpy array)
        y_pred: Predicted values (numpy array)
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate XGBoost models using k-fold cross validation')
    parser.add_argument('--target', type=str, default='all',
                       choices=['all', 'lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target variable to evaluate (default: all)')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def load_xgboost_models(target='all'):
    """
    Load trained XGBoost models.
    
    Args:
        target: Which target model(s) to load ('all' or specific target name)
        
    Returns:
        Dictionary of models or a single model, and feature list
    """
    models = {}
    
    # Define the XGBoost models directory
    xgboost_dir = PATHS.model_dir / 'xgboost'
    
    if target == 'all':
        # Load all target models
        for target_name in DATA.target_columns:
            model_path = xgboost_dir / f'xgboost_{target_name}_model.bin'
            if model_path.exists():
                model = xgb.Booster()  
                model.load_model(str(model_path))
                models[target_name] = model
            else:
                print(f"[WARNING] Model for {target_name} not found at {model_path}")
        
        # Also load feature list
        feature_path = xgboost_dir / f'xgboost_{DATA.target_columns[0]}_features.json'
        with open(feature_path, 'r') as f:
            feature_list = json.load(f)
        
        return models, feature_list
    else:
        # Load specific target model
        model_path = xgboost_dir / f'xgboost_{target}_model.bin'
        if model_path.exists():
            model = xgb.Booster()
            model.load_model(str(model_path))
            
            # Load feature list
            feature_path = xgboost_dir / f'xgboost_{target}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return model, feature_list
        else:
            raise FileNotFoundError(f"Model for {target} not found at {model_path}")

def plot_metrics(metrics_history, save_path):
    """
    Plot metrics across folds
    
    Args:
        metrics_history: Dictionary with lists of metrics per fold
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Get all metrics and folds
    metrics = list(metrics_history.keys())
    n_folds = len(metrics_history[metrics[0]])
    folds = list(range(1, n_folds + 1))
    
    # Plot each metric
    for metric in metrics:
        plt.plot(folds, metrics_history[metric], 'o-', label=metric)
    
    plt.xlabel('Fold')
    plt.ylabel('Error Value')
    plt.title('XGBoost Cross-Validation Metrics by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(folds)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved metrics plot to: {save_path}")

def evaluate_xgboost_model(model, X_df, y_true, target_idx=None):
    """
    Evaluate the XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_df: DataFrame with engineered features
        y_true: True target values
        target_idx: Index of the target being predicted
        
    Returns:
        Dictionary with evaluation metrics
    """
    dtest = xgb.DMatrix(X_df)
    y_pred = model.predict(dtest)
    
    if target_idx is not None:
        y_actual = y_true[:, target_idx]
        return calculate_metrics(y_actual, y_pred)
    else:
        # Multi-target evaluation
        metrics = {}
        for i, name in enumerate(DATA.target_columns):
            metrics.update(calculate_metrics(y_true[:, i], y_pred[:, i]))
        return metrics

def main():
    # Parse command line arguments
    args = parse_args()
    target_var = args.target
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"[INFO] Evaluating XGBoost model for target '{target_var}' with {args.k_folds}-fold cross validation")
    
    # 1. Load data
    print("[INFO] Loading sequence data...")
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    # Load scalers
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)
    
    # 2. Load models
    models, feature_list = load_xgboost_models(target_var)
    
    # 3. Engineer features
    print("[INFO] Engineering features from sequences...")
    feature_names = DATA.feature_columns
    X_train_df = engineer_features(X_train, feature_names)
    X_test_df = engineer_features(X_test, feature_names)
    
    # Make sure all expected features are present
    for feat in feature_list:
        if feat not in X_train_df.columns:
            X_train_df[feat] = 0
        if feat not in X_test_df.columns:
            X_test_df[feat] = 0
    
    # Keep only the features used during training
    X_train_df = X_train_df[feature_list]
    X_test_df = X_test_df[feature_list]
    
    # Flatten y data
    y_train_flat = y_train.squeeze(1)  # Remove seq_len dimension
    y_test_flat = y_test.squeeze(1)
    
    # Set up K-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Storage for metrics
    fold_metrics = {
        "MSE": [],
        "RMSE": [],
        "MAE": []
    }
    
    # K-fold cross validation
    start_time = time.time()
    
    if target_var == 'all':
        # Evaluate all target models
        for i, target_name in enumerate(DATA.target_columns):
            if target_name not in models:
                print(f"[WARNING] Skipping {target_name} - model not found")
                continue
                
            print(f"\n[INFO] Evaluating model for target '{target_name}'")
            target_fold_metrics = {
                "MSE": [],
                "RMSE": [],
                "MAE": []
            }
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
                print(f"  Fold {fold+1}/{args.k_folds}")
                
                # Split data for this fold
                X_fold_val = X_train_df.iloc[val_idx]
                y_fold_val = y_train_flat[val_idx]
                
                # Evaluate model on validation set
                fold_result = evaluate_xgboost_model(
                    models[target_name], X_fold_val, y_fold_val, i
                )
                
                print(f"    Fold {fold+1} results:")
                for metric, value in fold_result.items():
                    print(f"      {metric}: {value:.6f}")
                    target_fold_metrics[metric].append(value)
            
            # Calculate and print average metrics for this target
            print(f"\n[INFO] Average metrics for '{target_name}' across all folds:")
            for metric, values in target_fold_metrics.items():
                avg_value = np.mean(values)
                std_value = np.std(values)
                print(f"  {metric}: {avg_value:.6f} ± {std_value:.6f}")
                
            # Create visualizations directory if it doesn't exist
            vis_dir = PATHS.model_dir / "xgboost" / "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot metrics across folds for this target
            plot_metrics(
                target_fold_metrics, 
                vis_dir / f"{target_name}_cv_metrics.png"
            )
            
            # Evaluate on test set
            test_metrics = evaluate_xgboost_model(
                models[target_name], X_test_df, y_test_flat, i
            )
            
            print(f"\n[INFO] Test set evaluation results for '{target_name}':")
            for metric, value in test_metrics.items():
                print(f"  {metric}: {value:.6f}")
                
            # Save results to JSON file
            results = {
                "model_type": "xgboost",
                "target": target_name,
                "k_folds": args.k_folds,
                "cross_validation": {
                    metric: {
                        "values": [float(v) for v in values],
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    } for metric, values in target_fold_metrics.items()
                },
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
                "evaluation_time": time.time() - start_time
            }
            
            results_path = PATHS.model_dir / "xgboost" / f"{target_name}_evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"[INFO] Saved evaluation results to: {results_path}")
    else:
        # Evaluate specific target model
        target_idx = DATA.target_columns.index(target_var)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_df)):
            print(f"\n[INFO] Fold {fold+1}/{args.k_folds}")
            
            # Split data for this fold
            X_fold_val = X_train_df.iloc[val_idx]
            y_fold_val = y_train_flat[val_idx]
            
            # Evaluate model on validation set
            fold_result = evaluate_xgboost_model(
                models, X_fold_val, y_fold_val, target_idx
            )
            
            print(f"  Fold {fold+1} results:")
            for metric, value in fold_result.items():
                print(f"    {metric}: {value:.6f}")
                fold_metrics[metric].append(value)
        
        # Calculate and print average metrics
        print("\n[INFO] Average metrics across all folds:")
        for metric, values in fold_metrics.items():
            avg_value = np.mean(values)
            std_value = np.std(values)
            print(f"  {metric}: {avg_value:.6f} ± {std_value:.6f}")
        
        # Create visualizations directory if it doesn't exist
        vis_dir = PATHS.model_dir / "xgboost" / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot metrics across folds
        plot_metrics(fold_metrics, vis_dir / f"{target_var}_cv_metrics.png")
        
        # Evaluate on test set
        test_metrics = evaluate_xgboost_model(
            models, X_test_df, y_test_flat, target_idx
        )
        
        print("\n[INFO] Test set evaluation results:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Save results to JSON file
        results = {
            "model_type": "xgboost",
            "target": target_var,
            "k_folds": args.k_folds,
            "cross_validation": {
                metric: {
                    "values": [float(v) for v in values],
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                } for metric, values in fold_metrics.items()
            },
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "evaluation_time": time.time() - start_time
        }
        
        results_path = PATHS.model_dir / "xgboost" / f"{target_var}_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[INFO] Saved evaluation results to: {results_path}")

if __name__ == "__main__":
    main() 
