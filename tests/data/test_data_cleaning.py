"""
This module contains tests for the functions in the data module.

Tests cover the fetching, cleaning, preprocessing of Bitcoin data and adding technical indicators.
"""

import pandas as pd
import os
import joblib
import numpy as np

from src.data.data_cleaning import clean_data, normalize_data


def test_clean_data():
    """
    Test the clean_data function.
    """
    data = pd.DataFrame(
        {
            "Open": [1, 2, 2, 3],
            "High": [1, 2, 2, 3],
            "Low": [1, 2, 2, 3],
            "Close": [1, 2, 2, 3],
            "Volume": [1, 2, 2, 3],
        }
    )
    cleaned_data = clean_data(data)
    assert not cleaned_data.empty
    assert cleaned_data.shape[0] == 3  # Check that duplicates are removed


def test_normalize_data():
    """
    Test the normalize_data function.
    """
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [2, 3, 4, 5],
            "Low": [0, 1, 2, 3],
            "Close": [1, 2, 3, 4],
            "Volume": [100, 200, 300, 400],
        }
    )
    normalized_data, _ = normalize_data(data)

    # Check that the normalized data has the correct shape
    assert normalized_data.shape == data.shape

    # Check that the values are in the range [0, 1]
    assert normalized_data.min().min() >= 0
    assert normalized_data.max().max() <= 1


def test_normalize_data_with_path(tmpdir):
    """
    Test the normalize_data function with a path provided.
    """
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4],
            "High": [2, 3, 4, 5],
            "Low": [0, 1, 2, 3],
            "Close": [1, 2, 3, 4],
            "Volume": [100, 200, 300, 400],
        }
    )
    path = os.path.join(tmpdir, "scaler.pkl")
    normalized_data, scaler = normalize_data(data, path=path)

    # Load the scaler from the file
    loaded_scaler = joblib.load(path)

    # Check that the loaded scaler transforms data in the same way as the original scaler
    assert np.allclose(loaded_scaler.transform(data), scaler.transform(data))
