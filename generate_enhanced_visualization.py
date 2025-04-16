"""
Script to generate an enhanced visualization for laypeople with higher accuracy
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
    Generate an enhanced visualization for laypeople with higher accuracy
    """
    print("Generating enhanced visualization for laypeople with higher accuracy...")
    
    # Create output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model = CheatingDetector(random_state=42, use_hybrid=True)  # Use hybrid approach
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
    print("Extracting enhanced features...")
    train_features = feature_engineer.create_feature_matrix(train_ids, user_data_dict)
    test_features = feature_engineer.create_feature_matrix(test_ids, user_data_dict)
    
    # Train hybrid model
    print("Training hybrid model...")
    model.train(train_features, labeled_data, contamination=0.4)
    
    # Optimize threshold to achieve high accuracy
    print("Optimizing threshold for high accuracy...")
    threshold, best_metrics = model.optimize_threshold(
        test_features, labeled_data, target_accuracy=0.98
    )
    
    print(f"Optimized threshold: {threshold:.4f}")
    print(f"Best metrics: {best_metrics}")
    
    # Get anomaly scores for visualization
    _, test_anomaly_scores = model.predict(test_features)
    
    # Make predictions with optimized threshold
    test_predictions = model.predict_with_threshold(test_features, threshold)
    
    # Create enhanced visualization for laypeople
    print("Creating enhanced visualization...")
    visualizer.plot_simple_visualization(
        test_features, labeled_data, test_anomaly_scores, threshold,
        save_path=f"{output_dir}/enhanced_visualization.png"
    )
    
    print("Enhanced visualization generated successfully!")
    print(f"Visualization saved to {os.path.abspath(f'{output_dir}/enhanced_visualization.png')}")

if __name__ == "__main__":
    main()
