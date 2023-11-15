"""
This module provides a function for cleaning and preprocessing data.

Function:
- clean_data: Handles missing values, checks for duplicates, and normalizes the data.

This module uses pandas for data manipulation and sklearn.preprocessing for data normalization.
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def clean_data(data):
    """
    Cleans the data by handling missing values, checking for duplicates, and normalizing.

    :param data: A Pandas DataFrame with the data.
    :return: A cleaned and preprocessed DataFrame.
    """
    # Handle missing values
    data = data.dropna()

    # Check for duplicates
    data = data.drop_duplicates()

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data_scaled
