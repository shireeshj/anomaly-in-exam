"""
Evaluation Module for Cheating Detection Application

This module handles evaluating model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from typing import Dict, Tuple, List, Any


class ModelEvaluator:
    def __init__(self):
        """
        Initialize the ModelEvaluator.
        """
        pass

    def evaluate(self, true_labels: pd.DataFrame, predictions: np.ndarray,
                 user_ids: list) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            true_labels: DataFrame containing true labels
            predictions: Array of predictions (1: cheating, 0: clean)
            user_ids: List of user IDs corresponding to predictions

        Returns:
            Dictionary of evaluation metrics
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

        # Calculate metrics
        accuracy = accuracy_score(merged_df['true_label'], merged_df['predicted'])
        precision = precision_score(merged_df['true_label'], merged_df['predicted'])
        recall = recall_score(merged_df['true_label'], merged_df['predicted'])
        f1 = f1_score(merged_df['true_label'], merged_df['predicted'])

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(merged_df['true_label'], merged_df['predicted']).ravel()

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

        return metrics

    def print_evaluation_report(self, metrics: Dict[str, float]) -> None:
        """
        Print a formatted evaluation report.

        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\n===== Evaluation Report =====")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

        # Print additional metrics if available
        if 'roc_auc' in metrics:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"PR AUC: {metrics['pr_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

        # Calculate and print additional metrics
        total = metrics['true_positives'] + metrics['false_positives'] + metrics['true_negatives'] + metrics['false_negatives']
        print(f"\nTotal examples: {total}")

        # Calculate specificity (true negative rate)
        if metrics['true_negatives'] + metrics['false_positives'] > 0:
            specificity = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives'])
            print(f"Specificity: {specificity:.4f}")

        # Calculate positive predictive value (precision)
        if metrics['true_positives'] + metrics['false_positives'] > 0:
            ppv = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
            print(f"Positive Predictive Value: {ppv:.4f}")

        # Calculate negative predictive value
        if metrics['true_negatives'] + metrics['false_negatives'] > 0:
            npv = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_negatives'])
            print(f"Negative Predictive Value: {npv:.4f}")

        print("=============================\n")

    def get_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Args:
            model: Trained model
            feature_names: List of feature names

        Returns:
            DataFrame containing feature importance
        """
        # For Isolation Forest, we can use the feature importances
        # based on the average path length for each feature
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })

            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)

            return importance_df
        else:
            return pd.DataFrame({'feature': feature_names, 'importance': np.nan})

    def get_misclassified_examples(self, true_labels: pd.DataFrame, predictions: np.ndarray,
                                  user_ids: list, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get misclassified examples for analysis.

        Args:
            true_labels: DataFrame containing true labels
            predictions: Array of predictions (1: cheating, 0: clean)
            user_ids: List of user IDs corresponding to predictions
            features: DataFrame containing features

        Returns:
            Tuple of (false_positives, false_negatives)
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

        # Merge with features
        full_df = pd.merge(merged_df, features, on='elitmus_id')

        # Get false positives (predicted cheating but actually clean)
        false_positives = full_df[(full_df['predicted'] == 1) & (full_df['true_label'] == 0)]

        # Get false negatives (predicted clean but actually cheating)
        false_negatives = full_df[(full_df['predicted'] == 0) & (full_df['true_label'] == 1)]

        return false_positives, false_negatives

    def evaluate_with_scores(self, true_labels: pd.DataFrame, predictions: np.ndarray,
                         anomaly_scores: np.ndarray, user_ids: list) -> Dict[str, float]:
        """
        Evaluate model performance with anomaly scores for ROC and PR curves.

        Args:
            true_labels: DataFrame containing true labels
            predictions: Array of predictions (1: cheating, 0: clean)
            anomaly_scores: Array of anomaly scores
            user_ids: List of user IDs corresponding to predictions

        Returns:
            Dictionary of evaluation metrics including ROC and PR AUC
        """
        # Get basic metrics
        metrics = self.evaluate(true_labels, predictions, user_ids)

        # Create DataFrame with predictions and scores
        result_df = pd.DataFrame({
            'elitmus_id': user_ids,
            'predicted': predictions,
            'anomaly_score': anomaly_scores
        })

        # Merge with true labels
        merged_df = pd.merge(result_df, true_labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['true_label'] = (merged_df['label'] == 'cheating').astype(int)

        # Calculate ROC AUC
        roc_auc = roc_auc_score(merged_df['true_label'], merged_df['anomaly_score'])
        metrics['roc_auc'] = roc_auc

        # Calculate PR AUC (average precision)
        pr_auc = average_precision_score(merged_df['true_label'], merged_df['anomaly_score'])
        metrics['pr_auc'] = pr_auc

        return metrics

    def plot_advanced_metrics(self, true_labels: pd.DataFrame, predictions: np.ndarray,
                             anomaly_scores: np.ndarray, user_ids: list,
                             output_dir: str = 'output') -> None:
        """
        Plot advanced metrics like ROC curve, PR curve, and confusion matrix.

        Args:
            true_labels: DataFrame containing true labels
            predictions: Array of predictions (1: cheating, 0: clean)
            anomaly_scores: Array of anomaly scores
            user_ids: List of user IDs corresponding to predictions
            output_dir: Directory to save plots
        """
        import os

        # Create DataFrame with predictions and scores
        result_df = pd.DataFrame({
            'elitmus_id': user_ids,
            'predicted': predictions,
            'anomaly_score': anomaly_scores
        })

        # Merge with true labels
        merged_df = pd.merge(result_df, true_labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['true_label'] = (merged_df['label'] == 'cheating').astype(int)

        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(merged_df['true_label'], merged_df['anomaly_score'])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right")

        # Save ROC curve
        plt.savefig(os.path.join(output_dir, 'advanced_roc_curve.png'))
        plt.close()

        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(merged_df['true_label'], merged_df['anomaly_score'])
        pr_auc = average_precision_score(merged_df['true_label'], merged_df['anomaly_score'])

        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.legend(loc="lower left")

        # Save PR curve
        plt.savefig(os.path.join(output_dir, 'advanced_pr_curve.png'))
        plt.close()

        # Plot detailed confusion matrix
        cm = confusion_matrix(merged_df['true_label'], merged_df['predicted'])
        plt.figure(figsize=(10, 8))

        # Plot confusion matrix with percentages
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Clean', 'Cheating'],
                   yticklabels=['Clean', 'Cheating'])

        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=16)

        # Save confusion matrix
        plt.savefig(os.path.join(output_dir, 'advanced_confusion_matrix.png'))
        plt.close()
