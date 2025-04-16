"""
Feature Engineering Module for Cheating Detection Application

This module handles extracting features from the raw data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import datetime
from scipy import stats
from collections import Counter


class FeatureEngineer:
    def __init__(self):
        """
        Initialize the FeatureEngineer.
        """
        pass

    def extract_features(self, user_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Extract features from user data.

        Args:
            user_data: Dictionary containing question results and user activities for a user

        Returns:
            Dictionary of features
        """
        question_features = self._extract_question_features(user_data['question_results'])
        activity_features = self._extract_activity_features(user_data['user_activities'])

        # Combine all features
        features = {**question_features, **activity_features}

        return features

    def _extract_question_features(self, question_results: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from question results.

        Args:
            question_results: DataFrame containing question results for a user

        Returns:
            Dictionary of features
        """
        features = {}

        if len(question_results) == 0:
            # Default values if no question results
            return {
                'avg_time_per_question': 0,
                'correct_ratio': 0,
                'skipped_ratio': 0,
                'time_deviation': 0,
                'time_vs_avg_ratio': 0,
                'total_questions': 0,
                'answered_ratio': 0,
                'avg_time_correct': 0,
                'avg_time_incorrect': 0,
                'time_ratio_correct_incorrect': 0,
                'time_std_dev': 0,
                'time_variance': 0,
                'time_skewness': 0,
                'time_kurtosis': 0,
                'time_min_max_ratio': 0,
                'time_median': 0,
                'time_q1': 0,
                'time_q3': 0,
                'time_iqr': 0,
                'time_range': 0,
                'time_consistency_score': 0,
                'correct_streak_max': 0,
                'incorrect_streak_max': 0,
                'skipped_streak_max': 0,
                'answer_change_rate': 0,
                'time_efficiency_score': 0,
                'suspicious_pattern_score': 0
            }

        # Basic statistics
        features['total_questions'] = len(question_results)
        features['avg_time_per_question'] = question_results['seconds_taken'].mean()

        # Calculate ratios
        features['correct_ratio'] = question_results['correct'].mean() if 'correct' in question_results else 0
        features['skipped_ratio'] = question_results['skipped'].mean() if 'skipped' in question_results else 0

        # Calculate answered ratio (non-skipped questions)
        answered = (~question_results['skipped'].astype(bool)).sum() if 'skipped' in question_results else 0
        features['answered_ratio'] = answered / features['total_questions'] if features['total_questions'] > 0 else 0

        # Advanced time statistics
        if 'seconds_taken' in question_results:
            times = question_results['seconds_taken']

            # Statistical measures of time distribution
            features['time_std_dev'] = times.std()
            features['time_variance'] = times.var()
            features['time_skewness'] = stats.skew(times) if len(times) > 2 else 0
            features['time_kurtosis'] = stats.kurtosis(times) if len(times) > 2 else 0
            features['time_median'] = times.median()
            features['time_q1'] = times.quantile(0.25)
            features['time_q3'] = times.quantile(0.75)
            features['time_iqr'] = features['time_q3'] - features['time_q1']
            features['time_range'] = times.max() - times.min() if len(times) > 0 else 0

            # Min-max ratio (how much faster the fastest question is compared to the slowest)
            if times.max() > 0:
                features['time_min_max_ratio'] = times.min() / times.max()
            else:
                features['time_min_max_ratio'] = 0

            # Time consistency score (lower value means more consistent timing)
            features['time_consistency_score'] = features['time_std_dev'] / features['avg_time_per_question'] if features['avg_time_per_question'] > 0 else 0

            # Time efficiency score (how efficiently time is used compared to average)
            if 'avg_time' in question_results and 'correct' in question_results:
                # Calculate efficiency for each question (correct answers with less time than average get higher scores)
                question_results['efficiency'] = np.where(
                    question_results['correct'] == True,
                    question_results['avg_time'] / question_results['seconds_taken'],
                    question_results['seconds_taken'] / question_results['avg_time']
                )
                features['time_efficiency_score'] = question_results['efficiency'].mean()

        # Time-related features with average time comparison
        if 'seconds_taken' in question_results and 'avg_time' in question_results:
            # Time deviation from average
            question_results['time_deviation'] = question_results['seconds_taken'] - question_results['avg_time']
            features['time_deviation'] = question_results['time_deviation'].mean()

            # Ratio of time taken vs average time
            question_results['time_ratio'] = question_results['seconds_taken'] / question_results['avg_time']
            features['time_vs_avg_ratio'] = question_results['time_ratio'].mean()

            # Time for correct vs incorrect answers
            if 'correct' in question_results:
                correct_times = question_results[question_results['correct'] == True]['seconds_taken']
                incorrect_times = question_results[question_results['correct'] == False]['seconds_taken']

                features['avg_time_correct'] = correct_times.mean() if len(correct_times) > 0 else 0
                features['avg_time_incorrect'] = incorrect_times.mean() if len(incorrect_times) > 0 else 0

                # Ratio of time spent on correct vs incorrect
                if features['avg_time_incorrect'] > 0:
                    features['time_ratio_correct_incorrect'] = features['avg_time_correct'] / features['avg_time_incorrect']
                else:
                    features['time_ratio_correct_incorrect'] = 0

        # Pattern-based features
        if 'correct' in question_results:
            # Convert to list for streak analysis
            correct_list = question_results['correct'].fillna(False).astype(bool).tolist()
            skipped_list = question_results['skipped'].fillna(False).astype(bool).tolist()

            # Calculate max streaks
            features['correct_streak_max'] = self._max_streak(correct_list, True)
            features['incorrect_streak_max'] = self._max_streak(correct_list, False)
            features['skipped_streak_max'] = self._max_streak(skipped_list, True)

            # Calculate suspicious pattern score
            # This looks for patterns that might indicate cheating, such as:
            # - Very fast correct answers
            # - Unusual consistency in timing
            # - Perfect scores in difficult questions
            suspicious_score = 0

            # Check for unusually fast correct answers
            if 'avg_time' in question_results:
                fast_correct = ((question_results['correct'] == True) &
                               (question_results['seconds_taken'] < question_results['avg_time'] * 0.3)).sum()
                suspicious_score += fast_correct / features['total_questions'] if features['total_questions'] > 0 else 0

            # Check for unusual consistency in timing
            if features['time_consistency_score'] < 0.1 and features['total_questions'] > 5:
                suspicious_score += 0.5

            # Check for perfect scores in difficult questions
            if features['correct_ratio'] > 0.9 and features['total_questions'] > 5:
                suspicious_score += 0.3

            features['suspicious_pattern_score'] = suspicious_score

        return features

    def _max_streak(self, values: List[bool], target_value: bool) -> int:
        """
        Calculate the maximum streak of a target value in a list.

        Args:
            values: List of boolean values
            target_value: The value to look for streaks of

        Returns:
            Maximum streak length
        """
        max_streak = current_streak = 0

        for value in values:
            if value == target_value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _extract_activity_features(self, user_activities: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from user activities.

        Args:
            user_activities: DataFrame containing user activities

        Returns:
            Dictionary of features
        """
        features = {}

        if len(user_activities) == 0:
            # Default values if no activities
            return {
                'activity_count': 0,
                'unique_activities': 0,
                'login_count': 0,
                'view_to_answer_ratio': 0,
                'avg_time_between_activities': 0,
                'navigation_frequency': 0,
                'back_navigation_ratio': 0,
                'activity_entropy': 0,
                'time_between_std_dev': 0,
                'time_between_max': 0,
                'time_between_min': 0,
                'activity_sequence_entropy': 0,
                'repeated_view_ratio': 0,
                'answer_time_consistency': 0,
                'navigation_pattern_score': 0,
                'suspicious_timing_score': 0,
                'activity_burst_score': 0,
                'idle_time_ratio': 0
            }

        # Basic counts
        features['activity_count'] = len(user_activities)
        features['unique_activities'] = user_activities['activity'].nunique()

        # Count specific activities
        features['login_count'] = (user_activities['activity'] == 'login').sum()

        # Calculate view to answer ratio
        view_count = (user_activities['activity'] == 'viewed').sum()
        answer_count = (user_activities['activity'] == 'answered').sum()
        features['view_to_answer_ratio'] = view_count / answer_count if answer_count > 0 else 0

        # Navigation patterns
        next_count = (user_activities['activity'] == 'moved to next question').sum()
        back_count = (user_activities['activity'] == 'moved to previous question').sum()
        features['navigation_frequency'] = (next_count + back_count) / features['activity_count'] if features['activity_count'] > 0 else 0
        features['back_navigation_ratio'] = back_count / next_count if next_count > 0 else 0

        # Time between activities and advanced timing features
        if 'created_at' in user_activities.columns:
            try:
                user_activities['timestamp'] = pd.to_datetime(user_activities['created_at'])
                user_activities = user_activities.sort_values('timestamp')

                # Calculate time differences in seconds
                user_activities['time_diff'] = user_activities['timestamp'].diff().dt.total_seconds()

                # Filter out NaN values (first row)
                time_diffs = user_activities['time_diff'].dropna()

                if len(time_diffs) > 0:
                    # Basic time statistics
                    features['avg_time_between_activities'] = time_diffs.mean()
                    features['time_between_std_dev'] = time_diffs.std()
                    features['time_between_max'] = time_diffs.max()
                    features['time_between_min'] = time_diffs.min()

                    # Calculate idle time ratio (time spent not doing anything)
                    # Define idle time as time differences > 60 seconds
                    idle_time = time_diffs[time_diffs > 60].sum()
                    total_time = time_diffs.sum()
                    features['idle_time_ratio'] = idle_time / total_time if total_time > 0 else 0

                    # Calculate answer time consistency
                    # Get time differences for answer activities
                    answer_activities = user_activities[user_activities['activity'] == 'answered']
                    if len(answer_activities) > 1:
                        answer_activities['time_since_view'] = answer_activities.apply(
                            lambda row: self._time_since_last_view(row, user_activities), axis=1
                        )
                        answer_times = answer_activities['time_since_view'].dropna()
                        if len(answer_times) > 0:
                            features['answer_time_consistency'] = answer_times.std() / answer_times.mean() if answer_times.mean() > 0 else 0

                    # Calculate activity burst score (rapid succession of activities)
                    burst_count = (time_diffs < 1).sum()  # Activities less than 1 second apart
                    features['activity_burst_score'] = burst_count / len(time_diffs) if len(time_diffs) > 0 else 0

                    # Calculate suspicious timing score
                    suspicious_timing = 0

                    # Check for unusually consistent timing between activities
                    if features['time_between_std_dev'] < 0.5 and len(time_diffs) > 10:
                        suspicious_timing += 0.5

                    # Check for unusually fast answers
                    if 'answer_time_consistency' in features and features['answer_time_consistency'] < 0.2:
                        suspicious_timing += 0.5

                    features['suspicious_timing_score'] = suspicious_timing
            except Exception as e:
                # If there's an error, set default values
                features['avg_time_between_activities'] = 0
                features['time_between_std_dev'] = 0
                features['time_between_max'] = 0
                features['time_between_min'] = 0
                features['idle_time_ratio'] = 0
                features['answer_time_consistency'] = 0
                features['activity_burst_score'] = 0
                features['suspicious_timing_score'] = 0

        # Calculate activity entropy (measure of randomness in activities)
        activity_counts = user_activities['activity'].value_counts(normalize=True)
        features['activity_entropy'] = -(activity_counts * np.log2(activity_counts)).sum()

        # Calculate activity sequence entropy (measure of predictability in activity sequence)
        if len(user_activities) > 1:
            # Create pairs of consecutive activities
            activity_sequence = list(zip(user_activities['activity'].iloc[:-1], user_activities['activity'].iloc[1:]))
            sequence_counts = Counter(activity_sequence)
            sequence_probs = {seq: count/len(activity_sequence) for seq, count in sequence_counts.items()}
            features['activity_sequence_entropy'] = -sum(p * np.log2(p) for p in sequence_probs.values())
        else:
            features['activity_sequence_entropy'] = 0

        # Calculate repeated view ratio (viewing the same question multiple times)
        if 'question_result_id' in user_activities.columns:
            view_activities = user_activities[user_activities['activity'] == 'viewed']
            if len(view_activities) > 0:
                unique_questions_viewed = view_activities['question_result_id'].nunique()
                total_views = len(view_activities)
                features['repeated_view_ratio'] = 1 - (unique_questions_viewed / total_views) if total_views > 0 else 0
            else:
                features['repeated_view_ratio'] = 0
        else:
            features['repeated_view_ratio'] = 0

        # Calculate navigation pattern score
        # This looks for patterns that might indicate cheating, such as:
        # - Unusual navigation patterns (e.g., jumping around questions)
        # - Minimal time spent on questions
        # - Repeated views of the same question
        navigation_score = 0

        # Check for unusual back-and-forth navigation
        if features['back_navigation_ratio'] > 0.5:
            navigation_score += 0.3

        # Check for repeated views of questions
        if features['repeated_view_ratio'] > 0.3:
            navigation_score += 0.3

        # Check for minimal time spent on questions
        if features['avg_time_between_activities'] < 5 and features['activity_count'] > 10:
            navigation_score += 0.4

        features['navigation_pattern_score'] = navigation_score

        return features

    def _time_since_last_view(self, answer_row: pd.Series, activities: pd.DataFrame) -> float:
        """
        Calculate the time between an answer activity and the last view activity for the same question.

        Args:
            answer_row: Row containing an answer activity
            activities: DataFrame containing all activities

        Returns:
            Time difference in seconds, or NaN if no matching view activity found
        """
        if 'question_result_id' not in answer_row or pd.isna(answer_row['question_result_id']):
            return np.nan

        # Get the question ID
        question_id = answer_row['question_result_id']
        answer_time = answer_row['timestamp']

        # Find the most recent view activity for this question before the answer
        view_activities = activities[
            (activities['activity'] == 'viewed') &
            (activities['question_result_id'] == question_id) &
            (activities['timestamp'] < answer_time)
        ]

        if len(view_activities) == 0:
            return np.nan

        # Get the most recent view time
        last_view_time = view_activities['timestamp'].max()

        # Calculate time difference in seconds
        time_diff = (answer_time - last_view_time).total_seconds()

        return time_diff

    def create_feature_matrix(self, user_ids: List[int], user_data_dict: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create a feature matrix for multiple users.

        Args:
            user_ids: List of user IDs
            user_data_dict: Dictionary mapping user IDs to their data

        Returns:
            DataFrame containing features for all users
        """
        feature_list = []

        for user_id in user_ids:
            if user_id in user_data_dict:
                features = self.extract_features(user_data_dict[user_id])
                features['elitmus_id'] = user_id
                feature_list.append(features)

        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_list)

        return feature_df
