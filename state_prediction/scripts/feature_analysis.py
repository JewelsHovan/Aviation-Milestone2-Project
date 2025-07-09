#!/usr/bin/env python
"""
Script: feature_analysis.py
--------------------------
Performs feature importance analysis and ablation studies on the state prediction models.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from paths import DATA_DIR
from config import DataConfig, PATHS, DATA, MODEL, TRAIN, PathConfig
import sys
import fickling

sys.path.append(str(Path(__file__).parent.parent))
from models import get_model

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Feature importance analysis for state prediction models')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman', 'xgboost'],
                       default='transformer', help='Model architecture to use')
    parser.add_argument('--method', type=str, choices=['ablation', 'permutation', 'shap'],
                       default='ablation', help='Feature analysis method')
    return parser.parse_args()

def load_data():
    """Load train and test data"""
    # Load preprocessed data
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    # Load scalers
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = fickling.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = fickling.load(f)
    
    return X_train, y_train, X_test, y_test, X_scaler, y_scaler

def ablation_study(model, X_test, y_test, feature_names, device):
    """
    Perform ablation study by zeroing out each feature and measuring impact
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Create dataset for batch processing
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float().squeeze(1)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
    
    # Baseline performance with all features
    baseline_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            baseline_loss += loss.item() * X_batch.shape[0]
    
    baseline_loss /= len(test_dataset)
    print(f"Baseline loss: {baseline_loss:.6f}")
    
    # Test performance with each feature zeroed out
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        print(f"Testing feature: {feature_name} ({i+1}/{len(feature_names)})")
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # Create a copy of the batch and zero out the feature
                X_batch_ablated = X_batch.clone()
                X_batch_ablated[:, :, i] = 0
                
                # Move to device
                X_batch_ablated = X_batch_ablated.to(device)
                y_batch = y_batch.to(device)
                
                # Forward pass
                preds = model(X_batch_ablated)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * X_batch.shape[0]
                
                # Clear GPU cache to prevent memory buildup
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(test_dataset)
        importance = (avg_loss - baseline_loss) / baseline_loss * 100  # Percent increase in loss
        feature_importance[feature_name] = importance
        
        print(f"Feature {feature_name}: Ablated Loss = {avg_loss:.6f}, Importance = {importance:.2f}%")
    
    return feature_importance

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test, X_scaler, y_scaler = load_data()
    
    # Define feature names
    feature_names = DATA.feature_columns
    if not feature_names:
        feature_names = ["lon", "lat", "heading", "velocity", "vertrate", "geoaltitude"]
    
    # Set device
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")
    
    # Create model directory path based on model type
    model_dir = PATHS.model_dir / args.model
    vis_dir = model_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set the model type in the config
    MODEL.model_type = args.model
    
    # Get model parameters using the config helper method
    model_params = MODEL.get_model_params()
    
    # Load trained model
    model = get_model(args.model, **model_params).to(device)
    model.load_state_dict(torch.load(
        PATHS.get_model_weights_path(args.model),
        map_location=device
    ))
    
    if args.method == 'ablation':
        # Feature ablation study
        print(f"[INFO] Running ablation study for {args.model} model...")
        feature_importance = ablation_study(model, X_test, y_test, feature_names, device)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        importance_values = list(feature_importance.values())
        features = list(feature_importance.keys())
        
        # Sort by importance
        sorted_indices = np.argsort(importance_values)
        sorted_importance = [importance_values[i] for i in sorted_indices]
        sorted_features = [features[i] for i in sorted_indices]
        
        plt.barh(sorted_features, sorted_importance)
        plt.xlabel('Importance (% increase in loss when feature is ablated)')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for {args.model.capitalize()} Model')
        plt.tight_layout()
        plt.savefig(vis_dir / f"{args.model}_feature_importance.png", dpi=300)
        print(f"[INFO] Saved feature importance plot to: {vis_dir / f'{args.model}_feature_importance.png'}")
        
        # Save results to JSON
        results = {
            "model": args.model,
            "method": args.method,
            "feature_importance": feature_importance
        }
        
        with open(vis_dir / f"{args.model}_feature_importance.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"[INFO] Saved feature importance results to: {vis_dir / f'{args.model}_feature_importance.json'}")
    
    # Additional methods could be implemented here (permutation, SHAP)
    
    print("[INFO] Analysis completed.")

if __name__ == "__main__":
    main() 
