"""
Visualization Module for Cheating Detection Application

This module handles creating visual representations of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Dict, List, Tuple


class Visualizer:
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the Visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        # Get absolute path to output directory
        # If running from src directory, go up one level
        if os.path.basename(os.getcwd()) == 'src':
            self.output_dir = os.path.abspath(os.path.join(os.getcwd(), '..', output_dir))
        else:
            self.output_dir = os.path.abspath(output_dir)

        # Set style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_anomaly_scores(self, anomaly_scores: np.ndarray, true_labels: pd.DataFrame,
                           user_ids: list, threshold: float = None,
                           save_path: str = None) -> None:
        """
        Plot anomaly scores distribution.

        Args:
            anomaly_scores: Array of anomaly scores
            true_labels: DataFrame containing true labels
            user_ids: List of user IDs corresponding to anomaly scores
            threshold: Anomaly threshold (optional)
            save_path: Path to save the plot (optional)
        """
        # Create DataFrame with anomaly scores
        score_df = pd.DataFrame({
            'elitmus_id': user_ids,
            'anomaly_score': anomaly_scores
        })

        # Merge with true labels
        merged_df = pd.merge(score_df, true_labels, on='elitmus_id')

        plt.figure(figsize=(12, 8))

        # Plot distributions
        sns.histplot(data=merged_df, x='anomaly_score', hue='label',
                    kde=True, palette=['green', 'red'], alpha=0.6)

        # Add threshold line if provided
        if threshold is not None:
            plt.axvline(x=threshold, color='black', linestyle='--',
                       label=f'Threshold: {threshold:.3f}')

        plt.title('Distribution of Anomaly Scores by Class', fontsize=16)
        plt.xlabel('Anomaly Score', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Class')

        if save_path:
            plt.savefig(save_path)
            print(f"Saved anomaly score plot to {save_path}")

        plt.show()

    def plot_roc_curve(self, anomaly_scores: np.ndarray, true_labels: pd.DataFrame,
                      user_ids: list, save_path: str = None) -> None:
        """
        Plot ROC curve.

        Args:
            anomaly_scores: Array of anomaly scores
            true_labels: DataFrame containing true labels
            user_ids: List of user IDs corresponding to anomaly scores
            save_path: Path to save the plot (optional)
        """
        # Create DataFrame with anomaly scores
        score_df = pd.DataFrame({
            'elitmus_id': user_ids,
            'anomaly_score': anomaly_scores
        })

        # Merge with true labels
        merged_df = pd.merge(score_df, true_labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['true_label'] = (merged_df['label'] == 'cheating').astype(int)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(merged_df['true_label'], merged_df['anomaly_score'])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))

        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (area = {roc_auc:.3f})')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path)
            print(f"Saved ROC curve plot to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, true_labels: pd.DataFrame, predictions: np.ndarray,
                             user_ids: list, save_path: str = None) -> None:
        """
        Plot confusion matrix.

        Args:
            true_labels: DataFrame containing true labels
            predictions: Array of predictions (1: cheating, 0: clean)
            user_ids: List of user IDs corresponding to predictions
            save_path: Path to save the plot (optional)
        """
        # Create DataFrame with predictions
        pred_df = pd.DataFrame({
            'elitmus_id': user_ids,
            'predicted': predictions
        })

        # Merge with true labels
        merged_df = pd.merge(pred_df, true_labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['true_label'] = (merged_df['label'] == 'cheating').astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(merged_df['true_label'], merged_df['predicted'])

        plt.figure(figsize=(10, 8))

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Clean', 'Cheating'],
                   yticklabels=['Clean', 'Cheating'])

        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved confusion matrix plot to {save_path}")

        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame,
                               top_n: int = 10, save_path: str = None) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame containing feature importance
            top_n: Number of top features to show
            save_path: Path to save the plot (optional)
        """
        # Get top N features
        top_features = importance_df.head(top_n)

        plt.figure(figsize=(12, 8))

        # Plot feature importance
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')

        plt.title(f'Top {top_n} Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved feature importance plot to {save_path}")

        plt.show()

    def plot_feature_distributions(self, features: pd.DataFrame, true_labels: pd.DataFrame,
                                  top_features: List[str], save_path: str = None) -> None:
        """
        Plot distributions of top features by class.

        Args:
            features: DataFrame containing features
            true_labels: DataFrame containing true labels
            top_features: List of top features to plot
            save_path: Path to save the plot (optional)
        """
        # Merge features with true labels
        merged_df = pd.merge(features, true_labels, on='elitmus_id')

        # Create subplot grid
        n_features = len(top_features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        # Plot each feature
        for i, feature in enumerate(top_features):
            if i < len(axes):
                sns.kdeplot(data=merged_df, x=feature, hue='label',
                           palette=['green', 'red'], ax=axes[i])
                axes[i].set_title(feature)

        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved feature distributions plot to {save_path}")

        plt.show()

    def plot_2d_feature_space(self, features: pd.DataFrame, true_labels: pd.DataFrame,
                             feature1: str, feature2: str, anomaly_scores: np.ndarray = None,
                             save_path: str = None) -> None:
        """
        Plot 2D feature space with class labels.

        Args:
            features: DataFrame containing features
            true_labels: DataFrame containing true labels
            feature1: First feature to plot
            feature2: Second feature to plot
            anomaly_scores: Array of anomaly scores (optional)
            save_path: Path to save the plot (optional)
        """
        # Merge features with true labels
        merged_df = pd.merge(features, true_labels, on='elitmus_id')

        plt.figure(figsize=(12, 10))

        # Add anomaly scores if provided
        if anomaly_scores is not None and len(anomaly_scores) == len(merged_df):
            merged_df['anomaly_score'] = anomaly_scores
            scatter = plt.scatter(merged_df[feature1], merged_df[feature2],
                                 c=merged_df['anomaly_score'], cmap='viridis',
                                 alpha=0.7, s=100, edgecolors='k')
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            # Plot by class
            sns.scatterplot(data=merged_df, x=feature1, y=feature2,
                           hue='label', palette=['green', 'red'],
                           alpha=0.7, s=100, edgecolors='k')

        plt.title(f'Feature Space: {feature1} vs {feature2}', fontsize=16)
        plt.xlabel(feature1, fontsize=14)
        plt.ylabel(feature2, fontsize=14)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved 2D feature space plot to {save_path}")

        plt.show()

    def plot_simple_visualization(self, features: pd.DataFrame, true_labels: pd.DataFrame,
                                 anomaly_scores: np.ndarray, threshold: float,
                                 save_path: str = None) -> None:
        """
        Create a simple visualization for laypeople that clearly shows cheating vs clean cases.

        Args:
            features: DataFrame containing features
            true_labels: DataFrame containing true labels
            anomaly_scores: Array of anomaly scores
            threshold: Anomaly threshold
            save_path: Path to save the plot (optional)
        """
        # Create DataFrame with anomaly scores and predictions
        result_df = pd.DataFrame({
            'elitmus_id': features['elitmus_id'],
            'anomaly_score': anomaly_scores,
            'predicted': (anomaly_scores >= threshold).astype(int)
        })

        # Merge with true labels
        merged_df = pd.merge(result_df, true_labels, on='elitmus_id')

        # Sort by anomaly score for better visualization
        merged_df = merged_df.sort_values('anomaly_score')

        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

        # Plot 1: Scatter plot of anomaly scores (top left)
        ax1 = fig.add_subplot(gs[0, 0])

        # Create an index for the x-axis
        merged_df['index'] = range(len(merged_df))

        # Plot anomaly scores with larger points and clearer colors
        scatter1 = ax1.scatter(merged_df['index'], merged_df['anomaly_score'],
                             c=merged_df['label'].map({'clean': 'limegreen', 'cheating': 'crimson'}),
                             s=150, alpha=0.8, edgecolors='k')

        # Add threshold line
        ax1.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.3f}')

        # Add labels and title
        ax1.set_title('Cheating Detection Scores', fontsize=20)
        ax1.set_xlabel('User ID', fontsize=16)
        ax1.set_ylabel('Suspicion Score', fontsize=16)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                      markersize=15, label='Clean Users'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
                      markersize=15, label='Cheating Users'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                      label=f'Detection Threshold: {threshold:.3f}')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=14)

        # Add text annotations explaining the plot
        ax1.text(0.02, 0.02,
                "Each dot represents a user.\n" +
                "Users above the threshold are flagged as cheating.\n" +
                "Higher scores indicate more suspicious behavior.",
                transform=ax1.transAxes, fontsize=14, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # Plot 2: Pie chart of results (top right)
        ax2 = fig.add_subplot(gs[0, 1])

        # Calculate counts
        true_positive = ((merged_df['label'] == 'cheating') & (merged_df['predicted'] == 1)).sum()
        false_positive = ((merged_df['label'] == 'clean') & (merged_df['predicted'] == 1)).sum()
        true_negative = ((merged_df['label'] == 'clean') & (merged_df['predicted'] == 0)).sum()
        false_negative = ((merged_df['label'] == 'cheating') & (merged_df['predicted'] == 0)).sum()

        # Calculate accuracy
        accuracy = (true_positive + true_negative) / len(merged_df)

        # Create pie chart data
        labels = ['Correctly Identified Cheating', 'Falsely Flagged as Cheating',
                 'Correctly Identified Clean', 'Missed Cheating']
        sizes = [true_positive, false_positive, true_negative, false_negative]
        colors = ['darkred', 'lightcoral', 'darkgreen', 'lightgreen']
        explode = (0.1, 0.05, 0.1, 0.05)  # explode all slices

        # Plot pie chart
        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops={'fontsize': 14})

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')

        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax2.set_title(f'Detection Results (Accuracy: {accuracy:.1%})', fontsize=20)

        # Plot 3: Bar chart comparing clean vs cheating (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])

        # Count actual and predicted labels
        actual_counts = merged_df['label'].value_counts()
        predicted_counts = merged_df['predicted'].map({0: 'clean', 1: 'cheating'}).value_counts()

        # Create a DataFrame for plotting
        count_df = pd.DataFrame({
            'Actual': [actual_counts.get('clean', 0), actual_counts.get('cheating', 0)],
            'Predicted': [predicted_counts.get('clean', 0), predicted_counts.get('cheating', 0)]
        }, index=['Clean', 'Cheating'])

        # Plot grouped bar chart
        count_df.plot(kind='bar', ax=ax3, color=['limegreen', 'crimson'], width=0.7)
        ax3.set_title('Actual vs Predicted Counts', fontsize=20)
        ax3.set_xlabel('User Type', fontsize=16)
        ax3.set_ylabel('Number of Users', fontsize=16)

        # Add count labels on top of bars
        for container in ax3.containers:
            ax3.bar_label(container, fontsize=14, fontweight='bold')

        ax3.legend(fontsize=14)

        # Plot 4: Confusion matrix as a heatmap (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])

        # Create confusion matrix
        cm = np.array([
            [true_negative, false_positive],
            [false_negative, true_positive]
        ])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Clean', 'Cheating'],
                   yticklabels=['Clean', 'Cheating'],
                   annot_kws={"size": 20, "weight": "bold"})

        ax4.set_title('Confusion Matrix', fontsize=20)
        ax4.set_xlabel('Predicted Label', fontsize=16)
        ax4.set_ylabel('True Label', fontsize=16)

        # Add text explaining the confusion matrix
        ax4.text(1.05, 0.5,
                "True Negative: Clean users correctly identified\n" +
                "False Positive: Clean users wrongly flagged\n" +
                "False Negative: Cheating users missed\n" +
                "True Positive: Cheating users correctly caught",
                transform=ax4.transAxes, fontsize=14, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # Add overall title
        plt.suptitle('Cheating Detection Results', fontsize=24, fontweight='bold', y=0.98)

        # Add subtitle with accuracy information
        plt.figtext(0.5, 0.92,
                   f"Overall Accuracy: {accuracy:.1%} | Cheating Detection Rate: {true_positive/(true_positive+false_negative):.1%}",
                   ha="center", fontsize=18, bbox={"boxstyle":"round,pad=0.5", "facecolor":"white", "alpha":0.8})

        plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make room for suptitle

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved enhanced visualization to {save_path}")

        plt.show()
