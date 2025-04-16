"""
Synchronized Cheating Detection System

This script analyzes temporal patterns across users to detect synchronized answering behavior,
which may indicate group cheating where multiple users are receiving answers from a common source.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Create output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
print("Loading data...")
question_results = pd.read_csv("data/question_results.csv")
user_activities = pd.read_csv("data/user_activities.csv")
labeled_data = pd.read_csv("data/labeled.csv")

print(f"Loaded {len(question_results)} question results")
print(f"Loaded {len(user_activities)} user activities")
print(f"Loaded {len(labeled_data)} labeled examples")

# Fix column name typo in labeled data
if 'lable' in labeled_data.columns:
    labeled_data = labeled_data.rename(columns={'lable': 'label'})

# Check distribution of labels
label_counts = labeled_data['label'].value_counts()
print(f"Label distribution: {label_counts}")

# Convert timestamps to datetime objects
print("Processing timestamps...")
if 'created_at' in user_activities.columns:
    user_activities['timestamp'] = pd.to_datetime(user_activities['created_at'])
    user_activities = user_activities.sort_values('timestamp')

# Extract answer timestamps from user activities
answer_activities = user_activities[user_activities['activity'] == 'answered'].copy()

# Ensure we have question IDs for answers
if 'question_result_id' in answer_activities.columns:
    print("Analyzing synchronized answering patterns...")

    # Merge with question results to get the answers
    answer_data = pd.merge(
        answer_activities,
        question_results[['question_result_id', 'elitmus_id', 'question_id', 'correct']],
        on=['question_result_id', 'elitmus_id'],
        how='inner'
    )

    # Group by question_id to analyze synchronized answers
    question_groups = answer_data.groupby('question_id')

    # Store synchronized answering patterns
    sync_patterns = []

    # Define time window for synchronized answers (in seconds)
    time_window = 60  # 1 minute

    # Analyze each question
    for question_id, group in question_groups:
        # Skip questions with too few answers
        if len(group) < 5:
            continue

        # Sort by timestamp
        group = group.sort_values('timestamp')

        # Find clusters of answers within the time window
        clusters = []
        current_cluster = [group.iloc[0]]

        for i in range(1, len(group)):
            current_row = group.iloc[i]
            prev_row = group.iloc[i-1]

            time_diff = (current_row['timestamp'] - prev_row['timestamp']).total_seconds()

            if time_diff <= time_window:
                current_cluster.append(current_row)
            else:
                if len(current_cluster) >= 3:  # Only consider clusters with at least 3 users
                    clusters.append(current_cluster)
                current_cluster = [current_row]

        # Add the last cluster if it's large enough
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        # Analyze each cluster for answer similarity
        for cluster_idx, cluster in enumerate(clusters):
            cluster_df = pd.DataFrame(cluster)

            # Since we don't have the specific answer selected, we'll use the 'correct' field
            # to determine if users are answering correctly or incorrectly in sync
            correct_count = cluster_df['correct'].sum()
            incorrect_count = len(cluster_df) - correct_count

            # If there's a dominant pattern (more than 70% of users got the same result)
            dominant_count = max(correct_count, incorrect_count)
            if dominant_count / len(cluster_df) >= 0.7:
                dominant_answer = 'correct' if correct_count > incorrect_count else 'incorrect'

                # Create a record of this synchronized pattern
                sync_pattern = {
                    'question_id': question_id,
                    'cluster_id': f"{question_id}_{cluster_idx}",
                    'timestamp_start': cluster_df['timestamp'].min(),
                    'timestamp_end': cluster_df['timestamp'].max(),
                    'duration_seconds': (cluster_df['timestamp'].max() - cluster_df['timestamp'].min()).total_seconds(),
                    'num_users': len(cluster_df),
                    'dominant_answer': dominant_answer,
                    'dominant_answer_count': dominant_count,
                    'dominant_answer_percentage': (dominant_count / len(cluster_df)) * 100,
                    'is_correct': dominant_answer == 'correct',
                    'user_ids': cluster_df['elitmus_id'].tolist()
                }

                sync_patterns.append(sync_pattern)

    # Convert to DataFrame
    if sync_patterns:
        sync_patterns_df = pd.DataFrame(sync_patterns)
        print(f"Found {len(sync_patterns_df)} synchronized answering patterns")

        # Save the patterns to CSV
        sync_patterns_df.to_csv(f"{output_dir}/synchronized_patterns.csv", index=False)

        # Extract features for each user based on synchronized patterns
        user_sync_features = defaultdict(lambda: {
            'sync_pattern_count': 0,
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'correct_sync_ratio': 0,
            'avg_sync_duration': 0,
            'sync_answer_consistency': 0
        })

        # Count participation in synchronized patterns for each user
        for _, pattern in sync_patterns_df.iterrows():
            for user_id in pattern['user_ids']:
                user_sync_features[user_id]['sync_pattern_count'] += 1

                # Update max cluster size
                user_sync_features[user_id]['max_cluster_size'] = max(
                    user_sync_features[user_id]['max_cluster_size'],
                    pattern['num_users']
                )

                # Track if the synchronized answer was correct
                if pattern['is_correct']:
                    user_sync_features[user_id]['correct_sync_count'] = user_sync_features[user_id].get('correct_sync_count', 0) + 1

                # Track durations
                user_sync_features[user_id]['total_sync_duration'] = user_sync_features[user_id].get('total_sync_duration', 0) + pattern['duration_seconds']

        # Calculate averages and ratios
        for user_id, features in user_sync_features.items():
            if features['sync_pattern_count'] > 0:
                # Calculate average cluster size
                features['avg_cluster_size'] = features.get('avg_cluster_size', 0) / features['sync_pattern_count']

                # Calculate correct sync ratio
                features['correct_sync_ratio'] = features.get('correct_sync_count', 0) / features['sync_pattern_count']

                # Calculate average sync duration
                features['avg_sync_duration'] = features.get('total_sync_duration', 0) / features['sync_pattern_count']

            # Clean up temporary keys
            features.pop('correct_sync_count', None)
            features.pop('total_sync_duration', None)

        # Convert to DataFrame
        sync_features_df = pd.DataFrame.from_dict(user_sync_features, orient='index')
        sync_features_df.reset_index(inplace=True)
        sync_features_df.rename(columns={'index': 'elitmus_id'}, inplace=True)

        print(f"Extracted synchronized answering features for {len(sync_features_df)} users")

        # Merge with labeled data
        merged_df = pd.merge(sync_features_df, labeled_data, on='elitmus_id', how='inner')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['target'] = (merged_df['label'] == 'cheating').astype(int)

        # Analyze correlation between sync features and cheating
        # Select only numeric columns for correlation
        numeric_cols = merged_df.select_dtypes(include=['number']).columns
        correlation = merged_df[numeric_cols].corr()['target'].sort_values(ascending=False)
        print("\nCorrelation between synchronized features and cheating:")
        print(correlation)

        # Visualize synchronized patterns
        print("\nCreating visualizations...")

        # Plot 1: Distribution of synchronized pattern counts by label
        plt.figure(figsize=(12, 8))
        sns.histplot(
            data=merged_df,
            x='sync_pattern_count',
            hue='label',
            multiple='stack',
            palette={'cheating': 'crimson', 'clean': 'limegreen'},
            bins=10
        )
        plt.title('Distribution of Synchronized Pattern Counts by Label', fontsize=16)
        plt.xlabel('Number of Synchronized Patterns', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.savefig(f"{output_dir}/sync_pattern_distribution.png")

        # Plot 2: Scatter plot of sync_pattern_count vs avg_cluster_size
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            merged_df['sync_pattern_count'],
            merged_df['avg_cluster_size'],
            c=merged_df['target'].map({0: 'limegreen', 1: 'crimson'}),
            s=100, alpha=0.7, edgecolors='k'
        )
        plt.title('Synchronized Patterns vs. Average Cluster Size', fontsize=16)
        plt.xlabel('Number of Synchronized Patterns', fontsize=14)
        plt.ylabel('Average Cluster Size', fontsize=14)
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', markersize=10, label='Clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=10, label='Cheating')
        ])
        plt.savefig(f"{output_dir}/sync_pattern_scatter.png")

        # Plot 3: Heatmap of synchronized patterns over time
        if len(sync_patterns_df) > 0:
            # Create a timeline of synchronized patterns
            plt.figure(figsize=(15, 10))

            # Sort patterns by start time
            sorted_patterns = sync_patterns_df.sort_values('timestamp_start')

            # Create a colormap based on cluster size
            import matplotlib as mpl
            cmap = mpl.colormaps['viridis']
            norm = plt.Normalize(sorted_patterns['num_users'].min(), sorted_patterns['num_users'].max())
            colors = cmap(norm(sorted_patterns['num_users']))

            # Plot each pattern as a horizontal line
            for i, (_, pattern) in enumerate(sorted_patterns.iterrows()):
                start_time = pattern['timestamp_start']
                end_time = pattern['timestamp_end']
                plt.plot(
                    [start_time, end_time],
                    [i, i],
                    linewidth=5,
                    solid_capstyle='butt',
                    color=colors[i]
                )

                # Add a marker for correct/incorrect answers
                marker_color = 'green' if pattern['is_correct'] else 'red'
                plt.scatter(end_time, i, color=marker_color, s=100, zorder=5)

            # Add colorbar
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Number of Users in Cluster', fontsize=14)

            # Add legend for markers
            plt.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Correct Answer'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Incorrect Answer')
            ])

            plt.title('Timeline of Synchronized Answering Patterns', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Pattern ID', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sync_pattern_timeline.png")

        # Create a combined feature set for enhanced detection
        print("\nCombining synchronized features with individual features...")

        # Extract basic features (similar to high_accuracy_detector.py)
        user_features = {}

        # Process each user
        for user_id in labeled_data['elitmus_id']:
            # Get user data
            user_questions = question_results[question_results['elitmus_id'] == user_id]
            user_activities_data = user_activities[user_activities['elitmus_id'] == user_id]

            # Initialize features
            features = {}

            # Question features
            if len(user_questions) > 0:
                features['total_questions'] = len(user_questions)
                features['avg_time'] = user_questions['seconds_taken'].mean()
                features['correct_ratio'] = user_questions['correct'].mean() if 'correct' in user_questions else 0
                features['skipped_ratio'] = user_questions['skipped'].mean() if 'skipped' in user_questions else 0

                # Time consistency
                if len(user_questions) > 1:
                    features['time_std'] = user_questions['seconds_taken'].std()
                    features['time_consistency'] = features['time_std'] / features['avg_time'] if features['avg_time'] > 0 else 0
                else:
                    features['time_std'] = 0
                    features['time_consistency'] = 0
            else:
                features['total_questions'] = 0
                features['avg_time'] = 0
                features['correct_ratio'] = 0
                features['skipped_ratio'] = 0
                features['time_std'] = 0
                features['time_consistency'] = 0

            # Activity features
            if len(user_activities_data) > 0:
                features['activity_count'] = len(user_activities_data)
                features['unique_activities'] = user_activities_data['activity'].nunique()

                # Navigation patterns
                next_count = (user_activities_data['activity'] == 'moved to next question').sum()
                back_count = (user_activities_data['activity'] == 'moved to previous question').sum()
                features['navigation_ratio'] = back_count / next_count if next_count > 0 else 0
            else:
                features['activity_count'] = 0
                features['unique_activities'] = 0
                features['navigation_ratio'] = 0

            user_features[user_id] = features

        # Convert to DataFrame
        feature_df = pd.DataFrame.from_dict(user_features, orient='index')
        feature_df.reset_index(inplace=True)
        feature_df.rename(columns={'index': 'elitmus_id'}, inplace=True)

        # Merge with synchronized features
        combined_features = pd.merge(feature_df, sync_features_df, on='elitmus_id', how='left')

        # Fill NaN values with 0 (for users without synchronized patterns)
        combined_features.fillna(0, inplace=True)

        # Merge with labels
        merged_df = pd.merge(combined_features, labeled_data, on='elitmus_id', how='inner')

        # Convert labels to binary (1: cheating, 0: clean)
        merged_df['target'] = (merged_df['label'] == 'cheating').astype(int)

        # Split data
        X = merged_df.drop(['elitmus_id', 'label', 'target'], axis=1)
        y = merged_df['target']

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest classifier
        print("Training enhanced Random Forest classifier...")
        rf_classifier = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )

        rf_classifier.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = rf_classifier.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_classifier.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title('Top 15 Feature Importance (Including Synchronized Patterns)', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sync_feature_importance.png")

        # Create a comprehensive visualization
        plt.figure(figsize=(20, 16))

        # Create a 2x2 grid
        gs = plt.GridSpec(2, 2, height_ratios=[1.5, 1])

        # Plot 1: Scatter plot of sync_pattern_count vs avg_cluster_size (top left)
        ax1 = plt.subplot(gs[0, 0])
        scatter = ax1.scatter(
            merged_df['sync_pattern_count'],
            merged_df['avg_cluster_size'],
            c=merged_df['target'].map({0: 'limegreen', 1: 'crimson'}),
            s=100, alpha=0.7, edgecolors='k'
        )
        ax1.set_title('Synchronized Patterns vs. Average Cluster Size', fontsize=16)
        ax1.set_xlabel('Number of Synchronized Patterns', fontsize=14)
        ax1.set_ylabel('Average Cluster Size', fontsize=14)
        ax1.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', markersize=10, label='Clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=10, label='Cheating')
        ])

        # Plot 2: Confusion matrix (top right)
        ax2 = plt.subplot(gs[0, 1])
        cm_array = np.array([
            [tn, fp],
            [fn, tp]
        ])
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Clean', 'Cheating'],
                   yticklabels=['Clean', 'Cheating'],
                   annot_kws={"size": 20, "weight": "bold"}, ax=ax2)
        ax2.set_title('Confusion Matrix', fontsize=16)
        ax2.set_xlabel('Predicted Label', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=14)

        # Plot 3: Top features (bottom left)
        ax3 = plt.subplot(gs[1, 0])
        top_5_features = feature_importance.head(5)
        sns.barplot(x='importance', y='feature', data=top_5_features, palette='viridis', ax=ax3)
        ax3.set_title('Top 5 Most Important Features', fontsize=16)
        ax3.set_xlabel('Importance', fontsize=14)
        ax3.set_ylabel('Feature', fontsize=14)

        # Plot 4: Distribution of sync pattern counts (bottom right)
        ax4 = plt.subplot(gs[1, 1])
        sns.histplot(
            data=merged_df,
            x='sync_pattern_count',
            hue='label',
            multiple='stack',
            palette={'cheating': 'crimson', 'clean': 'limegreen'},
            bins=10,
            ax=ax4
        )
        ax4.set_title('Distribution of Synchronized Pattern Counts', fontsize=16)
        ax4.set_xlabel('Number of Synchronized Patterns', fontsize=14)
        ax4.set_ylabel('Count', fontsize=14)

        # Add overall title
        plt.suptitle('Synchronized Cheating Detection Results', fontsize=24, fontweight='bold', y=0.98)

        # Add subtitle with accuracy information
        plt.figtext(0.5, 0.92,
                   f"Overall Accuracy: {accuracy:.1%} | Cheating Detection Rate: {recall:.1%}",
                   ha="center", fontsize=18, bbox={"boxstyle":"round,pad=0.5", "facecolor":"white", "alpha":0.8})

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(f"{output_dir}/synchronized_cheating_detection.png")

        print(f"\nSynchronized cheating detection completed successfully!")
        print(f"Results saved to {output_dir}/synchronized_cheating_detection.png")
    else:
        print("No synchronized answering patterns found.")
else:
    print("Question result IDs not found in user activities. Cannot analyze synchronized patterns.")
