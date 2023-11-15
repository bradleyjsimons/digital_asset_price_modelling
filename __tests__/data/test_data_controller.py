"""
This module contains tests for the data_controller module.

The tests cover the main function which controls the data fetching, cleaning, and feature engineering process.

Functions:
- test_main: Tests the main function from the data_controller module.
"""

import pandas as pd
import pytest
from src.data.data_controller import main


@pytest.fixture
def mock_functions(mocker):
    """
    A pytest fixture that mocks the functions used in the main function.
    """
    df = pd.DataFrame({"Close": [1, 2, 3]})
    mocker.patch("src.data.data_controller.fetch_bitcoin_data", return_value=df)
    mocker.patch("src.data.data_controller.clean_data", return_value=df)
    mocker.patch(
        "src.data.data_controller.add_all_technical_indicators",
        return_value=df,
    )
    mocker.patch(
        "src.data.data_controller.add_blockchain_data",
        return_value=df,
    )
    mocker.patch("src.data.data_controller.extract_features", return_value=df)
    mocker.patch(
        "src.data.data_controller.train_test_split",
        return_value=(df, df, pd.Series([1, 2, 3]), pd.Series([1, 2, 3])),
    )


def test_main(mock_functions):
    """
    Test the main function from the data_controller module.

    This function tests the main function by checking that it runs without errors and returns a DataFrame.
    """
    result = main("2022-01-01", "2022-01-31")
    assert isinstance(result, pd.DataFrame)
