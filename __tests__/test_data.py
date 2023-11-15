"""
This module contains tests for the functions in the data module.

Tests cover the fetching, cleaning, preprocessing of Bitcoin data and adding technical indicators.
"""

import pytest
import pandas as pd
from src.data import (
    fetch_bitcoin_data,
    clean_data,
    add_all_technical_indicators,
    add_blockchain_data,
)


@pytest.fixture
def mock_yfinance(mocker):
    """
    A pytest fixture that mocks the yf.Ticker and yf.Ticker.history functions to return a mock response.
    """
    mock = mocker.patch("src.data.yf.Ticker")
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


@pytest.fixture
def mock_blockchain(mocker):
    """
    A pytest fixture that mocks the blockchain functions to return a mock response.
    """
    mocker.patch(
        "src.data.get_hash_rate_over_time",
        return_value=pd.DataFrame(
            {"hash-rate": [1, 2, 3]},
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        ),
    )
    mocker.patch(
        "src.data.get_avg_block_size",
        return_value=pd.DataFrame(
            {"avg-block-size": [1, 2, 3]},
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        ),
    )
    mocker.patch(
        "src.data.get_network_difficulty",
        return_value=pd.DataFrame(
            {"difficulty": [1, 2, 3]},
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        ),
    )
    mocker.patch(
        "src.data.get_miners_revenue",
        return_value=pd.DataFrame(
            {"miners-revenue": [1, 2, 3]},
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        ),
    )
    mocker.patch(
        "src.data.get_mempool_size",
        return_value=pd.DataFrame(
            {"mempool-size": [1, 2, 3]},
            index=pd.date_range(start="2022-01-01", end="2022-01-03"),
        ),
    )


@pytest.fixture
def mock_ta(mocker):
    """
    A pytest fixture that mocks the ta functions to return a mock response.
    """
    mocker.patch(
        "src.data.calculate_bollinger_bands",
        return_value=([1, 2, 3], [1, 2, 3], [1, 2, 3]),
    )
    mocker.patch(
        "src.data.calculate_stochastic_oscillator", return_value=([1, 2, 3], [1, 2, 3])
    )
    mocker.patch(
        "src.data.calculate_macd", return_value=([1, 2, 3], [1, 2, 3], [1, 2, 3])
    )
    mocker.patch("src.data.calculate_rsi", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_sma", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_ema", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_atr", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_macd_histogram", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_obv", return_value=[1, 2, 3])
    mocker.patch("src.data.calculate_cci", return_value=[1, 2, 3])


def test_fetch_bitcoin_data(mock_yfinance):
    """
    Test the fetch_bitcoin_data function.
    """
    df = fetch_bitcoin_data("2022-01-01", "2022-01-31")
    assert not df.empty
    assert "Dividends" not in df.columns
    assert "Stock Splits" not in df.columns


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


def test_add_all_technical_indicators(mock_ta):
    """
    Test the add_all_technical_indicators function.
    """
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 2, 3],
        }
    )
    data_with_indicators = add_all_technical_indicators(data)
    assert not data_with_indicators.empty
    assert "upper_bb" in data_with_indicators.columns
    assert "middle_bb" in data_with_indicators.columns
    assert "lower_bb" in data_with_indicators.columns
    assert "slowk" in data_with_indicators.columns
    assert "slowd" in data_with_indicators.columns
    assert "macd" in data_with_indicators.columns
    assert "macdsignal" in data_with_indicators.columns
    assert "macdhist" in data_with_indicators.columns
    assert "rsi" in data_with_indicators.columns
    assert "sma" in data_with_indicators.columns
    assert "ema" in data_with_indicators.columns
    assert "atr" in data_with_indicators.columns
    assert "macd_hist" in data_with_indicators.columns
    assert "obv" in data_with_indicators.columns
    assert "cci" in data_with_indicators.columns


def test_add_blockchain_data(mock_blockchain):
    """
    Test the add_blockchain_data function.
    """
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [1, 2, 3],
        },
        index=pd.date_range(start="2022-01-01", end="2022-01-03"),
    )
    data_with_blockchain = add_blockchain_data(data)
    assert not data_with_blockchain.empty
    assert "hash-rate" in data_with_blockchain.columns
    assert "avg-block-size" in data_with_blockchain.columns
    assert "difficulty" in data_with_blockchain.columns
    assert "miners-revenue" in data_with_blockchain.columns
    assert "mempool-size" in data_with_blockchain.columns
