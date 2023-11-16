"""
This module provides functions for cleaning and preprocessing data.

Functions:
- clean_data: Handles missing values by forward filling and checks for duplicates.
- normalize_data: Normalizes the data using MinMaxScaler from sklearn.preprocessing.

These functions use pandas for data manipulation and sklearn.preprocessing for data normalization.
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def clean_data(data):
    """
    Cleans the data by forward filling missing values, dropping any rows that still contains missing values,
    and checking for duplicates.

    :param data: A Pandas DataFrame with the data.
    :return: A cleaned and preprocessed DataFrame.
    """
    # ensure all column titles are type str
    data.columns = data.columns.astype(str)

    # Forward fill missing values
    data.ffill(inplace=True)

    # Drop any rows that still contain missing values
    data.dropna(inplace=True)

    # Check for duplicates
    data.drop_duplicates(inplace=True)

    return data


def normalize_data(data, scalar=None):
    """
    Normalizes the data using MinMaxScaler.

    This function uses the MinMaxScaler from sklearn.preprocessing to normalize the data.
    It scales and transforms the data, and then returns a new DataFrame with the same columns and index as the original data.

    :param data: A Pandas DataFrame with the data to be normalized.
    :param scalar: An optional MinMaxScaler instance. If not provided, a new MinMaxScaler will be created.
    :return: A new Pandas DataFrame with the normalized data.
    """
    if not scalar:
        scaler = MinMaxScaler()

    data_scaled = pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns, index=data.index
    )

    return data_scaled
