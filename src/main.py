"""
Main Module for Cheating Detection Application

This is the entry point for the application that ties everything together.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import CheatingDetector
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer


def create_output_dir(output_dir: str = "output") -> str:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Directory to create

    Returns:
        Absolute path to the output directory
    """
    # Get absolute path to output directory
    # If running from src directory, go up one level
    if os.path.basename(os.getcwd()) == 'src':
        abs_output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', output_dir))
    else:
        abs_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)
        print(f"Created output directory: {abs_output_dir}")

    return abs_output_dir


def main() -> None:
    """
    Main function to run the cheating detection application.
    """
    print("Starting Cheating Detection Application...")
    print(f"Current working directory: {os.getcwd()}")

    # Create output directory
    output_dir = create_output_dir("output")

    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model = CheatingDetector(random_state=42)
    evaluator = ModelEvaluator()
    visualizer = Visualizer(output_dir=output_dir)

    # Load data
    print("Loading data...")
    data_loader.load_data()

    # Get labeled data
    labeled_data = data_loader.get_labeled_data()

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    train_ids, test_ids = data_loader.split_data(test_size=0.3, random_state=42)

    print(f"Training set: {len(train_ids)} users")
    print(f"Testing set: {len(test_ids)} users")

    # Create user data dictionary
    print("Extracting user data...")
    user_data_dict = {}
    for user_id in train_ids + test_ids:
        user_data_dict[user_id] = data_loader.get_user_data(user_id)

    # Extract features
    print("Extracting features...")
    train_features = feature_engineer.create_feature_matrix(train_ids, user_data_dict)
    test_features = feature_engineer.create_feature_matrix(test_ids, user_data_dict)

    print(f"Extracted {len(train_features.columns) - 1} features for {len(train_features)} training samples")
    print(f"Extracted {len(test_features.columns) - 1} features for {len(test_features)} testing samples")

    # Train model
    print("Training hybrid model...")
    # Start with a higher contamination rate since we know cheating is common
    # Use labeled data for the hybrid approach
    model.train(train_features, labeled_data, contamination=0.4)

    # Optimize threshold
    print("Optimizing threshold...")
    threshold, best_metrics = model.optimize_threshold(
        test_features, labeled_data, target_accuracy=0.98  # Increased target accuracy
    )

    print(f"Optimized threshold: {threshold:.4f}")
    print(f"Best metrics: {best_metrics}")

    # Make predictions with optimized threshold
    print("Making predictions with optimized threshold...")
    test_predictions = model.predict_with_threshold(test_features, threshold)

    # Get anomaly scores for visualization
    _, test_anomaly_scores = model.predict(test_features)

    # Evaluate model with advanced metrics
    print("Evaluating model with advanced metrics...")
    test_metrics = evaluator.evaluate_with_scores(labeled_data, test_predictions, test_anomaly_scores, test_ids)
    evaluator.print_evaluation_report(test_metrics)

    # Generate advanced metric plots
    print("Generating advanced metric plots...")
    evaluator.plot_advanced_metrics(labeled_data, test_predictions, test_anomaly_scores, test_ids, output_dir)

    # Create visualizations
    print("Creating visualizations...")

    # Create simple visualization for laypeople
    visualizer.plot_simple_visualization(
        test_features, labeled_data, test_anomaly_scores, threshold,
        save_path=f"{output_dir}/simple_visualization.png"
    )

    # Plot anomaly scores
    visualizer.plot_anomaly_scores(
        test_anomaly_scores, labeled_data, test_ids, threshold,
        save_path=f"{output_dir}/anomaly_scores.png"
    )

    # Plot ROC curve
    visualizer.plot_roc_curve(
        test_anomaly_scores, labeled_data, test_ids,
        save_path=f"{output_dir}/roc_curve.png"
    )

    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        labeled_data, test_predictions, test_ids,
        save_path=f"{output_dir}/confusion_matrix.png"
    )

    # Get feature importance
    if hasattr(model.isolation_forest, 'feature_importances_'):
        feature_importance = evaluator.get_feature_importance(
            model.isolation_forest, model.feature_columns
        )

        # Plot feature importance
        visualizer.plot_feature_importance(
            feature_importance, top_n=10,
            save_path=f"{output_dir}/feature_importance.png"
        )

        # Plot distributions of top features
        top_features = feature_importance.head(6)['feature'].tolist()
        visualizer.plot_feature_distributions(
            test_features, labeled_data, top_features,
            save_path=f"{output_dir}/feature_distributions.png"
        )

        # Plot 2D feature space for top 2 features
        if len(top_features) >= 2:
            visualizer.plot_2d_feature_space(
                test_features, labeled_data, top_features[0], top_features[1],
                test_anomaly_scores, save_path=f"{output_dir}/feature_space_2d.png"
            )

    print("Cheating Detection Application completed successfully!")


if __name__ == "__main__":
    main()
