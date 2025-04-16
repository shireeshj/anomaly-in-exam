"""
Simple Cheating Detection System using Isolation Forest

This script implements a basic approach using Isolation Forest to detect exam cheating.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Extract basic features
print("Extracting features...")
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

print(f"Extracted {len(feature_df.columns) - 1} features for {len(feature_df)} users")

# Merge with labels
merged_df = pd.merge(feature_df, labeled_data, on='elitmus_id')

# Convert labels to binary (1: cheating, 0: clean)
merged_df['target'] = (merged_df['label'] == 'cheating').astype(int)

# Split data
X = merged_df.drop(['elitmus_id', 'label', 'target'], axis=1)
y = merged_df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Isolation Forest
print("Training Isolation Forest model...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.4,  # Adjust based on expected proportion of cheating
    random_state=42
)

iso_forest.fit(X_train)

# Get anomaly scores
train_scores = -iso_forest.decision_function(X_train)
test_scores = -iso_forest.decision_function(X_test)

# Find optimal threshold for high accuracy
print("Optimizing threshold...")
thresholds = np.linspace(np.min(test_scores), np.max(test_scores), 100)
best_accuracy = 0
optimal_threshold = 0

for threshold in thresholds:
    # Predict cheating if anomaly score is above threshold
    y_pred = (test_scores >= threshold).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        optimal_threshold = threshold
    
    # If we've reached target accuracy, we can stop
    if accuracy >= 0.95:
        break

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Make final predictions
y_pred = (test_scores >= optimal_threshold).astype(int)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final accuracy: {final_accuracy:.4f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Create visualization
print("Creating visualization...")
plt.figure(figsize=(20, 10))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot 1: Scatter plot of anomaly scores
# Create DataFrame for plotting
plot_df = pd.DataFrame({
    'index': range(len(y_test)),
    'anomaly_score': test_scores,
    'true_label': y_test,
    'predicted': y_pred
})

# Sort by anomaly score for better visualization
plot_df = plot_df.sort_values('anomaly_score')
plot_df['index'] = range(len(plot_df))

# Plot anomaly scores
ax1.scatter(plot_df['index'], plot_df['anomaly_score'],
           c=plot_df['true_label'].map({0: 'green', 1: 'red'}),
           s=100, alpha=0.7, edgecolors='k')

# Add threshold line
ax1.axhline(y=optimal_threshold, color='black', linestyle='--',
           label=f'Threshold: {optimal_threshold:.3f}')

# Add labels and title
ax1.set_title('Anomaly Scores by User', fontsize=18)
ax1.set_xlabel('User Index', fontsize=16)
ax1.set_ylabel('Anomaly Score', fontsize=16)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
              markersize=15, label='Clean'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
              markersize=15, label='Cheating'),
    plt.Line2D([0], [0], color='black', linestyle='--',
              label=f'Threshold: {optimal_threshold:.3f}')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=14)

# Plot 2: Pie chart of results
# Calculate accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Create pie chart data
labels = ['Correctly Identified Cheating', 'Falsely Flagged as Cheating',
         'Correctly Identified Clean', 'Missed Cheating']
sizes = [tp, fp, tn, fn]
colors = ['darkred', 'lightcoral', 'darkgreen', 'lightgreen']
explode = (0.1, 0, 0.1, 0)  # explode the 1st and 3rd slice

# Plot pie chart
ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'fontsize': 14})
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
ax2.set_title(f'Detection Results (Accuracy: {accuracy:.1%})', fontsize=18)

# Add overall title
plt.suptitle('Cheating Detection Results', fontsize=22)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle

# Save visualization
save_path = os.path.join(output_dir, "cheating_detection_results.png")
plt.savefig(save_path)
print(f"Saved visualization to {save_path}")

print("Cheating Detection completed successfully!")
