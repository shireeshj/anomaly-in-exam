# Anomaly in Exam

A machine learning application for detecting cheating in online exams using anomaly detection techniques with over 97% accuracy.

## Overview

This project implements a hybrid approach combining unsupervised and supervised learning methods to detect anomalous behavior in online exams that may indicate cheating. The system achieves over 97% accuracy in identifying cheating cases.

## Project Structure

- `src/`: Source code directory with modular components
- `data/`: Data directory
  - `labeled.csv`: Contains labeled data (cheating or clean)
  - `question_results.csv`: Contains user attempts on questions
  - `user_activities.csv`: Contains user navigation behavior
- `output/`: Output directory for visualizations and results
- `simple_cheating_detector.py`: Basic implementation using Isolation Forest
- `high_accuracy_detector.py`: Enhanced implementation with Random Forest
- `layman_visualization.py`: Creates intuitive visualizations for non-technical users
- `requirements.txt`: List of required packages

## Features

The application extracts various features from the data, including:

- Question-related features:
  - Average time spent per question
  - Ratio of correct answers
  - Ratio of skipped questions
  - Time deviation from average time per question
  - Time consistency metrics
  - Differences between time spent on correct vs. incorrect answers

- Activity-related features:
  - Frequency of navigation actions
  - Pattern of question viewing
  - Time between actions
  - Unusual navigation patterns
  - Activity entropy
  - Suspicious timing patterns

## Key Features for Detection

The most important features for detecting cheating are:
1. Total questions attempted
2. Activity count
3. Skipped ratio
4. Average time spent per question
5. Time ratio between correct and incorrect answers
6. Time consistency

## Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the high accuracy detector:
   ```
   python high_accuracy_detector.py
   ```

3. Generate layman-friendly visualization:
   ```
   python layman_visualization.py
   ```

4. View the results in the `output/` directory.

## Visualizations

The application generates several visualizations:

- Cheating detection scores for each user
- Distribution of results (correctly/incorrectly identified cases)
- Comparison of actual vs. predicted counts
- Confusion matrix with explanations
- Feature importance analysis

## Performance

The hybrid approach achieves:
- **Accuracy**: 97.22%
- **Precision**: 95.45%
- **Recall**: 100.00%
- **F1 Score**: 97.67%

This exceeds the original target of 95% accuracy and ensures that all cheating cases are correctly identified.

## Methods

The project implements two main approaches:
1. **Unsupervised Learning**: Isolation Forest for anomaly detection
2. **Supervised Learning**: Random Forest classifier for enhanced accuracy

The hybrid approach combines these methods to achieve superior results.

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

MIT
