"""
This module contains tests for the functions in the data module.

Tests cover the fetching, cleaning, preprocessing of Bitcoin data and adding technical indicators.
"""

import pytest
import pandas as pd

from src.data.data_cleaning import clean_data


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
