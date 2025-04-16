"""
Model Module for Cheating Detection Application

This module implements the Isolation Forest model for anomaly detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Dict, List, Any, Union


class CheatingDetector:
    def __init__(self, random_state: int = 42, use_hybrid: bool = True):
        """
        Initialize the CheatingDetector.

        Args:
            random_state: Random seed for reproducibility
            use_hybrid: Whether to use a hybrid approach (combining unsupervised and supervised methods)
        """
        self.random_state = random_state
        self.use_hybrid = use_hybrid
        self.isolation_forest = None
        self.supervised_model = None
        self.hybrid_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.threshold = None

    def train(self, features: pd.DataFrame, labels: pd.DataFrame = None, contamination: float = 0.1) -> None:
        """
        Train the model using either just Isolation Forest or a hybrid approach.

        Args:
            features: DataFrame containing features
            labels: DataFrame containing true labels (required for hybrid approach)
            contamination: Expected proportion of anomalies in the dataset
        """
        # Store feature columns (excluding ID column)
        self.feature_columns = [col for col in features.columns if col != 'elitmus_id']

        # Scale features
        X = self.scaler.fit_transform(features[self.feature_columns])

        # Initialize and train the Isolation Forest model
        self.isolation_forest = IsolationForest(
            n_estimators=200,  # Increased from 100
            max_samples='auto',
            contamination=contamination,
            max_features=0.8,  # Use a subset of features
            bootstrap=True,    # Use bootstrapping
            n_jobs=-1,         # Use all available cores
            random_state=self.random_state
        )

        self.isolation_forest.fit(X)

        # If using hybrid approach and labels are provided
        if self.use_hybrid and labels is not None:
            # Merge features with labels
            merged_df = pd.merge(features, labels, on='elitmus_id')

            # Convert labels to binary (1: cheating, 0: clean)
            y = (merged_df['label'] == 'cheating').astype(int)

            # Get anomaly scores from Isolation Forest
            anomaly_scores = -self.isolation_forest.decision_function(X)

            # Add anomaly scores as a feature
            X_with_scores = np.column_stack((X, anomaly_scores))

            # Train a Random Forest classifier
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )

            # Train an SVM classifier
            svm = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )

            # Train a Neural Network classifier
            nn = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )

            # Create a voting classifier
            self.supervised_model = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('svm', svm),
                    ('nn', nn)
                ],
                voting='soft',  # Use probability estimates
                weights=[3, 1, 1]  # Give more weight to Random Forest
            )

            # Train the supervised model
            self.supervised_model.fit(X_with_scores, y)

            print(f"Hybrid model trained on {len(features)} samples with {len(self.feature_columns) + 1} features")
        else:
            print(f"Isolation Forest model trained on {len(features)} samples with {len(self.feature_columns)} features")

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using the trained model.

        Args:
            features: DataFrame containing features

        Returns:
            Tuple of (predictions, anomaly_scores)
            - predictions: 1 for normal, -1 for anomaly (Isolation Forest) or 1 for cheating, 0 for clean (hybrid)
            - anomaly_scores: Anomaly scores (higher means more anomalous)
        """
        if self.isolation_forest is None:
            raise ValueError("Model has not been trained yet")

        # Ensure features have the same columns as training data
        missing_cols = set(self.feature_columns) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in test data: {missing_cols}")

        # Scale features
        X = self.scaler.transform(features[self.feature_columns])

        # Get anomaly scores from Isolation Forest (negative of decision function)
        # Higher score means more anomalous
        anomaly_scores = -self.isolation_forest.decision_function(X)

        # If using hybrid approach and supervised model is trained
        if self.use_hybrid and self.supervised_model is not None:
            # Add anomaly scores as a feature
            X_with_scores = np.column_stack((X, anomaly_scores))

            # Get predictions from supervised model (1: cheating, 0: clean)
            predictions = self.supervised_model.predict(X_with_scores)

            # Get probability estimates for the positive class (cheating)
            probabilities = self.supervised_model.predict_proba(X_with_scores)[:, 1]

            # Use probabilities as refined anomaly scores
            anomaly_scores = probabilities
        else:
            # Get predictions from Isolation Forest (1: normal, -1: anomaly)
            predictions = self.isolation_forest.predict(X)

            # Convert to binary (1: cheating, 0: clean) if threshold is set
            if self.threshold is not None:
                predictions = (anomaly_scores >= self.threshold).astype(int)

        return predictions, anomaly_scores

    def optimize_threshold(self, features: pd.DataFrame, labels: pd.DataFrame,
                           target_accuracy: float = 0.98) -> Tuple[float, Dict[str, float]]:
        """
        Optimize the anomaly threshold to achieve target accuracy.

        Args:
            features: DataFrame containing features
            labels: DataFrame containing true labels
            target_accuracy: Target accuracy to achieve (default increased to 98%)

        Returns:
            Tuple of (optimal_threshold, metrics)
        """
        # Get anomaly scores
        _, anomaly_scores = self.predict(features)

        # Merge with true labels
        result_df = pd.DataFrame({
            'elitmus_id': features['elitmus_id'],
            'anomaly_score': anomaly_scores
        })

        merged_df = pd.merge(result_df, labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['true_label'] = (merged_df['label'] == 'cheating').astype(int)

        # Try different thresholds to find optimal one
        # Use more fine-grained thresholds (200 instead of 100)
        thresholds = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores), 200)
        best_accuracy = 0
        optimal_threshold = 0
        best_metrics = {}

        for threshold in thresholds:
            # Predict cheating if anomaly score is above threshold
            predicted_labels = (merged_df['anomaly_score'] >= threshold).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(merged_df['true_label'], predicted_labels)

            # Calculate other metrics
            precision = precision_score(merged_df['true_label'], predicted_labels, zero_division=0)
            recall = recall_score(merged_df['true_label'], predicted_labels, zero_division=0)
            f1 = f1_score(merged_df['true_label'], predicted_labels, zero_division=0)

            # Use a weighted combination of metrics to find the best threshold
            # This helps balance precision and recall
            combined_score = (2 * accuracy) + precision + recall

            if combined_score > best_accuracy:
                best_accuracy = combined_score
                optimal_threshold = threshold

                best_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

            # If we've reached target accuracy, we can stop
            if accuracy >= target_accuracy:
                break

        # Store the optimal threshold
        self.threshold = optimal_threshold

        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Best metrics: Accuracy={best_metrics['accuracy']:.4f}, "
              f"Precision={best_metrics['precision']:.4f}, "
              f"Recall={best_metrics['recall']:.4f}, "
              f"F1={best_metrics['f1_score']:.4f}")

        return optimal_threshold, best_metrics

    def predict_with_threshold(self, features: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Predict using a specific anomaly threshold.

        Args:
            features: DataFrame containing features
            threshold: Anomaly threshold (if None, uses the stored optimal threshold)

        Returns:
            Array of predictions (1: cheating, 0: clean)
        """
        # Use the provided threshold or the stored one
        threshold_to_use = threshold if threshold is not None else self.threshold

        if threshold_to_use is None:
            raise ValueError("No threshold provided and no optimal threshold stored. "
                             "Call optimize_threshold first or provide a threshold.")

        # If using hybrid approach and supervised model is trained, use its predictions directly
        if self.use_hybrid and self.supervised_model is not None:
            predictions, _ = self.predict(features)
            return predictions
        else:
            # Otherwise, use Isolation Forest with threshold
            _, anomaly_scores = self.predict(features)

            # Predict cheating if anomaly score is above threshold
            predictions = (anomaly_scores >= threshold_to_use).astype(int)

            return predictions

    def tune_hyperparameters(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Tune hyperparameters for the supervised model using grid search.

        Args:
            features: DataFrame containing features
            labels: DataFrame containing true labels

        Returns:
            Dictionary of best hyperparameters
        """
        if not self.use_hybrid:
            print("Hyperparameter tuning is only available for hybrid approach.")
            return {}

        # Merge features with labels
        merged_df = pd.merge(features, labels, on='elitmus_id')

        # Convert labels to binary (1: cheating, 0: clean)
        y = (merged_df['label'] == 'cheating').astype(int)

        # Scale features
        X = self.scaler.transform(features[self.feature_columns])

        # Get anomaly scores from Isolation Forest
        anomaly_scores = -self.isolation_forest.decision_function(X)

        # Add anomaly scores as a feature
        X_with_scores = np.column_stack((X, anomaly_scores))

        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Create a Random Forest classifier
        rf = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Fit grid search
        grid_search.fit(X_with_scores, y)

        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")

        return best_params
