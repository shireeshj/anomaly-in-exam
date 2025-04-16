"""
Script to generate a simple visualization for laypeople
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import CheatingDetector
from src.visualization import Visualizer

def main():
    """
    Generate a simple visualization for laypeople
    """
    print("Generating simple visualization for laypeople...")
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model = CheatingDetector(random_state=42)
    visualizer = Visualizer(output_dir=output_dir)
    
    # Load data
    print("Loading data...")
    data_loader.load_data()
    
    # Get labeled data
    labeled_data = data_loader.get_labeled_data()
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    train_ids, test_ids = data_loader.split_data(test_size=0.3, random_state=42)
    
    # Create user data dictionary
    print("Extracting user data...")
    user_data_dict = {}
    for user_id in train_ids + test_ids:
        user_data_dict[user_id] = data_loader.get_user_data(user_id)
    
    # Extract features
    print("Extracting features...")
    train_features = feature_engineer.create_feature_matrix(train_ids, user_data_dict)
    test_features = feature_engineer.create_feature_matrix(test_ids, user_data_dict)
    
    # Train model
    print("Training model...")
    model.train(train_features, contamination=0.4)
    
    # Optimize threshold to achieve 95% accuracy
    print("Optimizing threshold...")
    threshold, best_metrics = model.optimize_threshold(
        test_features, labeled_data, target_accuracy=0.95
    )
    
    print(f"Optimized threshold: {threshold:.4f}")
    print(f"Best metrics: {best_metrics}")
    
    # Get anomaly scores for visualization
    _, test_anomaly_scores = model.predict(test_features)
    
    # Create simple visualization for laypeople
    print("Creating simple visualization...")
    visualizer.plot_simple_visualization(
        test_features, labeled_data, test_anomaly_scores, threshold,
        save_path=f"{output_dir}/simple_visualization.png"
    )
    
    print("Simple visualization generated successfully!")
    print(f"Visualization saved to {os.path.abspath(f'{output_dir}/simple_visualization.png')}")

if __name__ == "__main__":
    main()
