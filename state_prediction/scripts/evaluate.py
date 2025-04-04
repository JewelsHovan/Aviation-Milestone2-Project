#!/usr/bin/env python
"""
Script: evaluate.py
------------------
Evaluates trained models using k-fold cross validation and calculates
multiple error metrics including MSE, RMSE, and MAE.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import matplotlib
import fickling

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import argparse
import json
import time
import os

# Add parent directory to path for model imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import PATHS, DATA, MODEL, TRAIN, INFERENCE, TransformerConfig, LSTMConfig, FFNNConfig

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
    parser = argparse.ArgumentParser(description='Evaluate model using k-fold cross validation')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def load_config(model_path):
    """Load model configuration from JSON file"""
    if not model_path.exists():
        raise FileNotFoundError(f"No config file found at {model_path}")
    
    with open(model_path, 'r') as f:
        config = json.load(f)
    return config

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
    plt.title('Cross-Validation Metrics by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(folds)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved metrics plot to: {save_path}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Evaluating {args.model} model with {args.k_folds}-fold cross validation")
    
    # Load model configuration for specific model type
    config = load_config(PATHS.get_model_config_path(args.model))
    
    # Verify model type matches
    if config["model_type"] != args.model:
        raise ValueError(f"Requested model type '{args.model}' doesn't match saved model type '{config['model_type']}'")
    
    # Update MODEL config with saved values
    MODEL.model_type = config["model_type"]
    MODEL.input_dim = config["input_dim"]
    model_specific_config = config[f"{MODEL.model_type}_config"]
    
    # Update the specific model configuration
    if MODEL.model_type == "transformer":
        MODEL.transformer = TransformerConfig(**model_specific_config)
    elif MODEL.model_type == "lstm":
        MODEL.lstm = LSTMConfig(**model_specific_config)
    elif MODEL.model_type == "ffnn":
        MODEL.ffnn = FFNNConfig(**model_specific_config)

    # 1. Load scalers
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)

    # 2. Load training data for cross-validation
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).squeeze(1)
    
    # Set up K-fold cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Storage for metrics
    fold_metrics = {
        "MSE": [],
        "RMSE": [],
        "MAE": []
    }
    
    # Model evaluation function
    def evaluate_model(model, data_loader, criterion, device):
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # Convert to numpy arrays for metric calculation
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_preds)
        metrics["loss"] = total_loss / len(data_loader)
        
        return metrics
    
    # K-fold cross validation
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n[INFO] Fold {fold+1}/{args.k_folds}")
        
        # Create data loaders for this fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=TRAIN.batch_size,
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=TRAIN.batch_size,
            sampler=val_sampler
        )
        
        # 3. Build model
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
        else:
            raise ValueError(f"Unknown model type: {MODEL.model_type}")
        
        model = get_model(MODEL.model_type, **model_params).to(device)
        
        # Load pre-trained weights
        model.load_state_dict(torch.load(
            PATHS.get_model_weights_path(args.model),
            map_location=device
        ))
        
        criterion = nn.MSELoss()
        
        # Evaluate model on validation set
        fold_result = evaluate_model(model, val_loader, criterion, device)
        print(f"  Fold {fold+1} results:")
        for metric, value in fold_result.items():
            print(f"    {metric}: {value:.6f}")
            if metric in fold_metrics:
                fold_metrics[metric].append(value)
    
    # Calculate and print average metrics
    print("\n[INFO] Average metrics across all folds:")
    for metric, values in fold_metrics.items():
        avg_value = np.mean(values)
        std_value = np.std(values)
        print(f"  {metric}: {avg_value:.6f} ± {std_value:.6f}")
    
    # Create visualizations directory if it doesn't exist
    vis_dir = PATHS.model_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot metrics across folds
    plot_metrics(fold_metrics, vis_dir / f"{args.model}_cv_metrics.png")
    
    # Also evaluate on test set for final validation
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.float32)).to(device).squeeze(1)
    
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
    
    # Build final model for test evaluation
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
    else:
        raise ValueError(f"Unknown model type: {MODEL.model_type}")
    
    final_model = get_model(MODEL.model_type, **model_params).to(device)
    final_model.load_state_dict(torch.load(
        PATHS.get_model_weights_path(args.model),
        map_location=device
    ))
    
    # Evaluate on test set
    test_metrics = evaluate_model(final_model, test_loader, criterion, device)
    
    print("\n[INFO] Test set evaluation results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Save results to JSON file
    results = {
        "model_type": MODEL.model_type,
        "k_folds": args.k_folds,
        "cross_validation": {
            metric: {
                "values": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            } for metric, values in fold_metrics.items()
        },
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "evaluation_time": time.time() - start_time
    }
    
    results_path = PATHS.model_dir / f"{args.model}_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[INFO] Saved evaluation results to: {results_path}")

if __name__ == "__main__":
    main() 
