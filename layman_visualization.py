"""
Layman-friendly Visualization for Cheating Detection Results

This script creates an intuitive visualization of the high accuracy cheating detection results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

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

# Extract features
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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

# Make predictions
y_pred_proba = rf_classifier.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

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

# Create enhanced visualization for laypeople
print("Creating layman-friendly visualization...")

# Create figure with 2x2 subplots
fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

# Plot 1: Scatter plot of prediction probabilities (top left)
ax1 = fig.add_subplot(gs[0, 0])

# Create DataFrame for plotting
plot_df = pd.DataFrame({
    'index': range(len(y_test)),
    'probability': y_pred_proba,
    'true_label': y_test,
    'predicted': y_pred
})

# Sort by probability for better visualization
plot_df = plot_df.sort_values('probability')
plot_df['index'] = range(len(plot_df))

# Plot probabilities with larger points and clearer colors
ax1.scatter(plot_df['index'], plot_df['probability'],
           c=plot_df['true_label'].map({0: 'limegreen', 1: 'crimson'}),
           s=150, alpha=0.8, edgecolors='k')

# Add threshold line
ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=2,
           label='Detection Threshold: 0.5')

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
              label='Detection Threshold: 0.5')
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

# Create pie chart data
labels = ['Correctly Identified Cheating', 'Falsely Flagged as Cheating',
         'Correctly Identified Clean', 'Missed Cheating']
sizes = [tp, fp, tn, fn]
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
actual_counts = pd.Series({0: (y_test == 0).sum(), 1: (y_test == 1).sum()})
predicted_counts = pd.Series({0: (y_pred == 0).sum(), 1: (y_pred == 1).sum()})

# Create a DataFrame for plotting
count_df = pd.DataFrame({
    'Actual': [actual_counts.get(0, 0), actual_counts.get(1, 0)],
    'Predicted': [predicted_counts.get(0, 0), predicted_counts.get(1, 0)]
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
cm_array = np.array([
    [tn, fp],
    [fn, tp]
])

# Plot heatmap
sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False,
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
           f"Overall Accuracy: {accuracy:.1%} | Cheating Detection Rate: {recall:.1%}",
           ha="center", fontsize=18, bbox={"boxstyle":"round,pad=0.5", "facecolor":"white", "alpha":0.8})

plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make room for suptitle

# Save visualization
save_path = os.path.join(output_dir, "layman_friendly_visualization.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved layman-friendly visualization to {save_path}")

# Feature importance analysis
print("\nTop 10 most important features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print(feature_importance.head(10))

print("\nVisualization completed successfully!")
