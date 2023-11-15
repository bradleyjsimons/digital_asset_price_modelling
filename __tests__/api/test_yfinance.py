"""
This module contains tests for the yfinance API functions in the src.api.yfinance module.

The tests cover the fetching of Bitcoin data from Yahoo Finance.

Functions:
- test_fetch_bitcoin_data: Tests the fetch_bitcoin_data function from the src.api.yfinance module.
"""


import pandas as pd
import pytest
from src.api.yfinance import fetch_bitcoin_data


@pytest.fixture
def mock_yfinance(mocker):
    """
    A pytest fixture that mocks the yf.Ticker and yf.Ticker.history functions to return a mock response.
    """
    mock = mocker.patch("yfinance.Ticker")
    mock.return_value.history.return_value = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 2, 3],
            "Dividends": [0, 0, 0],
            "Stock Splits": [0, 0, 0],
        }
    )
    return mock


def test_fetch_bitcoin_data(mock_yfinance):
    """
    Test the fetch_bitcoin_data function.
    """
    df = fetch_bitcoin_data("2022-01-01", "2022-01-31")
    assert not df.empty
    assert "Dividends" not in df.columns
    assert "Stock Splits" not in df.columns
