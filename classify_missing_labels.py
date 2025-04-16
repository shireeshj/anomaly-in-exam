"""
Classify Users with Missing Labels

This script identifies elitmus_ids that have missing labels in the labeled.csv file
and classifies them as cheating or clean cases using our trained model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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

# Identify users with missing labels
missing_label_mask = labeled_data['label'].isna() | (labeled_data['label'] == '')
users_with_missing_labels = labeled_data[missing_label_mask]['elitmus_id'].tolist()
users_with_valid_labels = labeled_data[~missing_label_mask]['elitmus_id'].tolist()

print(f"Found {len(users_with_missing_labels)} users with missing labels")
print(f"Found {len(users_with_valid_labels)} users with valid labels")

# Extract features for all users
print("Extracting features for all users...")
user_features = {}

# Process each user
for user_id in set(users_with_missing_labels + users_with_valid_labels):
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
            
        # Time for correct vs incorrect answers
        if 'correct' in user_questions:
            correct_times = user_questions[user_questions['correct'] == True]['seconds_taken']
            incorrect_times = user_questions[user_questions['correct'] == False]['seconds_taken']
            
            features['avg_time_correct'] = correct_times.mean() if len(correct_times) > 0 else 0
            features['avg_time_incorrect'] = incorrect_times.mean() if len(incorrect_times) > 0 else 0
            
            # Ratio of time spent on correct vs incorrect
            if features['avg_time_incorrect'] > 0:
                features['time_ratio_correct_incorrect'] = features['avg_time_correct'] / features['avg_time_incorrect']
            else:
                features['time_ratio_correct_incorrect'] = 0
    else:
        features['total_questions'] = 0
        features['avg_time'] = 0
        features['correct_ratio'] = 0
        features['skipped_ratio'] = 0
        features['time_std'] = 0
        features['time_consistency'] = 0
        features['avg_time_correct'] = 0
        features['avg_time_incorrect'] = 0
        features['time_ratio_correct_incorrect'] = 0
    
    # Activity features
    if len(user_activities_data) > 0:
        features['activity_count'] = len(user_activities_data)
        features['unique_activities'] = user_activities_data['activity'].nunique()
        
        # Navigation patterns
        next_count = (user_activities_data['activity'] == 'moved to next question').sum()
        back_count = (user_activities_data['activity'] == 'moved to previous question').sum()
        features['navigation_ratio'] = back_count / next_count if next_count > 0 else 0
        
        # View to answer ratio
        view_count = (user_activities_data['activity'] == 'viewed').sum()
        answer_count = (user_activities_data['activity'] == 'answered').sum()
        features['view_to_answer_ratio'] = view_count / answer_count if answer_count > 0 else 0
    else:
        features['activity_count'] = 0
        features['unique_activities'] = 0
        features['navigation_ratio'] = 0
        features['view_to_answer_ratio'] = 0
    
    user_features[user_id] = features

# Convert to DataFrame
feature_df = pd.DataFrame.from_dict(user_features, orient='index')
feature_df.reset_index(inplace=True)
feature_df.rename(columns={'index': 'elitmus_id'}, inplace=True)

# Add synchronized features if available
try:
    # Process timestamps
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
            
            # Extract features for each user based on synchronized patterns
            user_sync_features = {}
            
            # Count participation in synchronized patterns for each user
            for _, pattern in sync_patterns_df.iterrows():
                for user_id in pattern['user_ids']:
                    if user_id not in user_sync_features:
                        user_sync_features[user_id] = {
                            'sync_pattern_count': 0,
                            'avg_cluster_size': 0,
                            'max_cluster_size': 0,
                            'correct_sync_ratio': 0,
                            'avg_sync_duration': 0,
                            'correct_sync_count': 0,
                            'total_sync_duration': 0
                        }
                    
                    user_sync_features[user_id]['sync_pattern_count'] += 1
                    
                    # Update max cluster size
                    user_sync_features[user_id]['max_cluster_size'] = max(
                        user_sync_features[user_id]['max_cluster_size'], 
                        pattern['num_users']
                    )
                    
                    # Track if the synchronized answer was correct
                    if pattern['is_correct']:
                        user_sync_features[user_id]['correct_sync_count'] += 1
                    
                    # Track durations
                    user_sync_features[user_id]['total_sync_duration'] += pattern['duration_seconds']
            
            # Calculate averages and ratios
            for user_id, features in user_sync_features.items():
                if features['sync_pattern_count'] > 0:
                    # Calculate average cluster size
                    features['avg_cluster_size'] = features['max_cluster_size'] / features['sync_pattern_count']
                    
                    # Calculate correct sync ratio
                    features['correct_sync_ratio'] = features['correct_sync_count'] / features['sync_pattern_count']
                    
                    # Calculate average sync duration
                    features['avg_sync_duration'] = features['total_sync_duration'] / features['sync_pattern_count']
                
                # Clean up temporary keys
                features.pop('correct_sync_count', None)
                features.pop('total_sync_duration', None)
            
            # Convert to DataFrame
            sync_features_df = pd.DataFrame.from_dict(user_sync_features, orient='index')
            sync_features_df.reset_index(inplace=True)
            sync_features_df.rename(columns={'index': 'elitmus_id'}, inplace=True)
            
            print(f"Extracted synchronized answering features for {len(sync_features_df)} users")
            
            # Merge with feature_df
            feature_df = pd.merge(feature_df, sync_features_df, on='elitmus_id', how='left')
            
            # Fill NaN values with 0 (for users without synchronized patterns)
            feature_df.fillna(0, inplace=True)
except Exception as e:
    print(f"Error extracting synchronized features: {e}")
    print("Continuing without synchronized features...")

# Split into training and prediction datasets
training_features = feature_df[feature_df['elitmus_id'].isin(users_with_valid_labels)]
prediction_features = feature_df[feature_df['elitmus_id'].isin(users_with_missing_labels)]

print(f"Prepared features for {len(training_features)} training users and {len(prediction_features)} prediction users")

# Merge training features with labels
labeled_data_valid = labeled_data[~missing_label_mask]
training_features = pd.merge(training_features, labeled_data_valid, on='elitmus_id')

# Convert labels to binary (1: cheating, 0: clean)
training_features['target'] = (training_features['label'] == 'cheating').astype(int)

# Prepare training data
X_train = training_features.drop(['elitmus_id', 'label', 'target'], axis=1)
y_train = training_features['target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Prepare prediction data
X_pred = prediction_features.drop(['elitmus_id'], axis=1)
X_pred_scaled = scaler.transform(X_pred)

# Train Random Forest classifier
print("Training Random Forest classifier...")
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

# Make predictions for users with missing labels
print("Predicting labels for users with missing labels...")
pred_proba = rf_classifier.predict_proba(X_pred_scaled)[:, 1]
pred_labels = (pred_proba >= 0.5).astype(int)

# Create results DataFrame
results_df = pd.DataFrame({
    'elitmus_id': prediction_features['elitmus_id'],
    'cheating_probability': pred_proba,
    'predicted_label': ['cheating' if p == 1 else 'clean' for p in pred_labels]
})

# Sort by cheating probability (descending)
results_df = results_df.sort_values('cheating_probability', ascending=False)

# Save results to CSV
results_df.to_csv(f"{output_dir}/missing_label_predictions.csv", index=False)

print(f"Saved predictions for {len(results_df)} users with missing labels to {output_dir}/missing_label_predictions.csv")

# Print summary
cheating_count = (results_df['predicted_label'] == 'cheating').sum()
clean_count = (results_df['predicted_label'] == 'clean').sum()

print(f"\nPrediction Summary:")
print(f"Cheating: {cheating_count} users ({cheating_count/len(results_df)*100:.1f}%)")
print(f"Clean: {clean_count} users ({clean_count/len(results_df)*100:.1f}%)")

# Create visualizations
print("\nCreating visualizations...")

# Plot 1: Distribution of cheating probabilities
plt.figure(figsize=(12, 8))
sns.histplot(
    data=results_df, 
    x='cheating_probability', 
    hue='predicted_label',
    multiple='stack',
    palette={'cheating': 'crimson', 'clean': 'limegreen'},
    bins=20
)
plt.title('Distribution of Cheating Probabilities for Users with Missing Labels', fontsize=16)
plt.xlabel('Probability of Cheating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.legend(title='Predicted Label')
plt.savefig(f"{output_dir}/missing_label_probability_distribution.png")

# Plot 2: Top 20 most likely cheating users
if cheating_count > 0:
    top_cheating = results_df[results_df['predicted_label'] == 'cheating'].head(min(20, cheating_count))
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top_cheating['elitmus_id'].astype(str), top_cheating['cheating_probability'], color='crimson')
    plt.title('Top Most Likely Cheating Users (Missing Labels)', fontsize=16)
    plt.xlabel('Probability of Cheating', fontsize=14)
    plt.ylabel('User ID', fontsize=14)
    plt.xlim(0, 1)
    plt.grid(axis='x', alpha=0.3)

    # Add probability values to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_cheating_missing_labels.png")

# Plot 3: Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 15 Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/missing_label_feature_importance.png")

# Create a complete labeled dataset with predictions
complete_labeled_df = labeled_data.copy()

# Add predictions for users with missing labels
for _, row in results_df.iterrows():
    complete_labeled_df.loc[
        complete_labeled_df['elitmus_id'] == row['elitmus_id'], 
        'label'
    ] = row['predicted_label']

# Save the complete labeled dataset
complete_labeled_df.to_csv(f"{output_dir}/complete_labeled_data.csv", index=False)
print(f"Saved complete labeled dataset to {output_dir}/complete_labeled_data.csv")

print("\nClassification of users with missing labels completed successfully!")
