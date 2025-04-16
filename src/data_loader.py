"""
Data Loader Module for Cheating Detection Application

This module handles loading and preprocessing the data from CSV files.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader with the directory containing the data files.

        Args:
            data_dir: Directory containing the data files
        """
        # Get the absolute path to the data directory
        # If running from src directory, go up one level
        if os.path.basename(os.getcwd()) == 'src':
            self.data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', data_dir))
        else:
            self.data_dir = os.path.abspath(data_dir)

        self.question_results = None
        self.user_activities = None
        self.labeled_data = None

    def load_data(self) -> None:
        """
        Load all data files into memory.
        """
        # Load question results
        self.question_results = pd.read_csv(f"{self.data_dir}/question_results.csv")

        # Load user activities
        self.user_activities = pd.read_csv(f"{self.data_dir}/user_activities.csv")

        # Load labeled data
        self.labeled_data = pd.read_csv(f"{self.data_dir}/labeled.csv")

        # Fix column name typo in labeled data if it exists
        if 'lable' in self.labeled_data.columns:
            self.labeled_data = self.labeled_data.rename(columns={'lable': 'label'})

        print(f"Loaded {len(self.question_results)} question results")
        print(f"Loaded {len(self.user_activities)} user activities")
        print(f"Loaded {len(self.labeled_data)} labeled examples")

    def get_user_ids(self) -> list:
        """
        Get a list of all unique user IDs.

        Returns:
            List of unique user IDs
        """
        return self.labeled_data['elitmus_id'].unique().tolist()

    def get_labeled_data(self) -> pd.DataFrame:
        """
        Get the labeled data.

        Returns:
            DataFrame containing labeled data
        """
        return self.labeled_data

    def get_user_data(self, elitmus_id: int) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific user.

        Args:
            elitmus_id: The elitmus ID of the user

        Returns:
            Dictionary containing question results and user activities for the user
        """
        user_question_results = self.question_results[self.question_results['elitmus_id'] == elitmus_id]
        user_activities = self.user_activities[self.user_activities['elitmus_id'] == elitmus_id]

        return {
            'question_results': user_question_results,
            'user_activities': user_activities
        }

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[list, list]:
        """
        Split the labeled data into training and testing sets.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_ids, test_ids)
        """
        from sklearn.model_selection import train_test_split

        user_ids = self.get_user_ids()
        train_ids, test_ids = train_test_split(
            user_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labeled_data.set_index('elitmus_id').loc[user_ids]['label']
        )

        return train_ids, test_ids
