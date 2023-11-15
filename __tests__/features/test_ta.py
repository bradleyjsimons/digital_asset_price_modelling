"""
This module contains tests for the technical analysis functions in the ta module.

Tests cover the calculation of various technical indicators.
"""

import pytest
import pandas as pd
from src.features.ta import (
    calculate_bollinger_bands,
    calculate_stochastic_oscillator,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_atr,
    calculate_macd_histogram,
    calculate_obv,
    calculate_cci,
)


@pytest.fixture
def mock_data():
    """
    A pytest fixture that returns a mock DataFrame with 'High', 'Low', 'Close', and 'Volume' columns.
    """
    data = pd.DataFrame(
        {
            "High": [1, 2, 3, 4, 5],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Close": [0.75, 1.75, 2.75, 3.75, 4.75],
            "Volume": [100, 200, 300, 400, 500],
        }
    )
    return data


def test_calculate_bollinger_bands(mocker, mock_data):
    """
    Test the calculate_bollinger_bands function.
    """
    # Mock the talib.BBANDS function to return fixed values
    mocker.patch("talib.BBANDS", return_value=(1, 2, 3))
    upper, middle, lower = calculate_bollinger_bands(mock_data["Close"])
    assert upper == 1
    assert middle == 2
    assert lower == 3


def test_calculate_stochastic_oscillator(mocker, mock_data):
    """
    Test the calculate_stochastic_oscillator function.
    """
    # Mock the talib.STOCH function to return fixed values
    mocker.patch("talib.STOCH", return_value=(1, 2))
    slowk, slowd = calculate_stochastic_oscillator(mock_data)
    assert slowk == 1
    assert slowd == 2


def test_calculate_macd(mocker, mock_data):
    """
    Test the calculate_macd function.
    """
    # Mock the talib.MACD function to return fixed values
    mocker.patch("talib.MACD", return_value=(1, 2, 3))
    macd, signal, hist = calculate_macd(mock_data["Close"])
    assert macd == 1
    assert signal == 2
    assert hist == 3


def test_calculate_rsi(mocker, mock_data):
    """
    Test the calculate_rsi function.
    """
    # Mock the talib.RSI function to return a fixed value
    mocker.patch("talib.RSI", return_value=1)
    rsi = calculate_rsi(mock_data["Close"])
    assert rsi == 1


def test_calculate_sma(mocker, mock_data):
    """
    Test the calculate_sma function.
    """
    # Mock the talib.SMA function to return a fixed value
    mocker.patch("talib.SMA", return_value=1)
    sma = calculate_sma(mock_data["Close"])
    assert sma == 1


def test_calculate_ema(mocker, mock_data):
    """
    Test the calculate_ema function.
    """
    # Mock the talib.EMA function to return a fixed value
    mocker.patch("talib.EMA", return_value=1)
    ema = calculate_ema(mock_data["Close"])
    assert ema == 1


def test_calculate_atr(mocker, mock_data):
    """
    Test the calculate_atr function.
    """
    # Mock the talib.ATR function to return a fixed value
    mocker.patch("talib.ATR", return_value=1)
    atr = calculate_atr(mock_data)
    assert atr == 1


def test_calculate_macd_histogram(mocker, mock_data):
    """
    Test the calculate_macd_histogram function.
    """
    # Mock the talib.MACD function to return fixed values
    mocker.patch("talib.MACD", return_value=(1, 2, 3))
    hist = calculate_macd_histogram(mock_data["Close"])
    assert hist == 3


def test_calculate_obv(mocker, mock_data):
    """
    Test the calculate_obv function.
    """
    # Mock the talib.OBV function to return a fixed value
    mocker.patch("talib.OBV", return_value=1)
    obv = calculate_obv(mock_data)
    assert obv == 1


def test_calculate_cci(mocker, mock_data):
    """
    Test the calculate_cci function.
    """
    # Mock the talib.CCI function to return a fixed value
    mocker.patch("talib.CCI", return_value=1)
    cci = calculate_cci(mock_data)
    assert cci == 1
