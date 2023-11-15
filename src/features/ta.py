"""
This module provides functions for calculating various technical indicators using the TA-Lib library.

Functions:
- calculate_bollinger_bands: Calculates Bollinger Bands.
- calculate_stochastic_oscillator: Calculates the Stochastic Oscillator.
- calculate_macd: Calculates the Moving Average Convergence Divergence (MACD).
- calculate_rsi: Calculates the Relative Strength Index (RSI).
- calculate_sma: Calculates the Simple Moving Average (SMA).
- calculate_ema: Calculates the Exponential Moving Average (EMA).
- calculate_atr: Calculates the Average True Range (ATR).
- calculate_macd_histogram: Calculates the MACD Histogram.
- calculate_obv: Calculates On Balance Volume (OBV).
- calculate_cci: Calculates the Commodity Channel Index (CCI).

Each function takes a Pandas DataFrame or Series with price data as input and returns the calculated indicator. 
The DataFrame or Series should contain 'Close' prices, and for some indicators, 'High', 'Low', and 'Volume' data 
are also required. The time period for the calculation can be specified for each indicator.
"""

import talib


def calculate_bollinger_bands(data, window=20):
    """
    Calculate Bollinger Bands.

    :param data: A Pandas Series or DataFrame with the price data.
    :param window: The number of periods to use for the calculation.
    :return: A DataFrame with the Bollinger Bands.
    """
    upper, middle, lower = talib.BBANDS(data, timeperiod=window)
    return upper, middle, lower


def calculate_stochastic_oscillator(
    data, fastk_period=14, slowk_period=3, slowd_period=3
):
    """
    Calculate the Stochastic Oscillator.

    :param data: A Pandas DataFrame with 'High', 'Low', and 'Close' columns.
    :param fastk_period: The time period for the fast %K.
    :param slowk_period: The time period for the slow %K.
    :param slowd_period: The time period for the slow %D.
    :return: Two DataFrames with the slow %K and slow %D of the Stochastic Oscillator.
    """
    slowk, slowd = talib.STOCH(
        data["High"],
        data["Low"],
        data["Close"],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return slowk, slowd


def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD).

    :param data: A Pandas Series or DataFrame with the price data.
    :param fastperiod: The short-term EMA period.
    :param slowperiod: The long-term EMA period.
    :param signalperiod: The signal line EMA period.
    :return: A DataFrame with the MACD line, the signal line, and the MACD histogram.
    """
    macd, signal, hist = talib.MACD(
        data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    return macd, signal, hist


def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).

    :param data: A Pandas Series or DataFrame with the price data.
    :param period: The number of periods to use for the calculation.
    :return: A DataFrame with the RSI.
    """
    rsi = talib.RSI(data, timeperiod=period)
    return rsi


def calculate_sma(data, period=30):
    """
    Calculate the Simple Moving Average (SMA).

    :param data: A Pandas Series or DataFrame with the price data.
    :param period: The number of periods to use for the calculation.
    :return: A DataFrame with the SMA.
    """
    sma = talib.SMA(data, timeperiod=period)
    return sma


def calculate_ema(data, period=30):
    """
    Calculate the Exponential Moving Average (EMA).

    :param data: A Pandas Series or DataFrame with the price data.
    :param period: The number of periods to use for the calculation.
    :return: A DataFrame with the EMA.
    """
    ema = talib.EMA(data, timeperiod=period)
    return ema


def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR).

    :param data: A Pandas DataFrame with the high, low, and close price data.
    :param period: The number of periods to use for the calculation.
    :return: A Series with the ATR.
    """
    atr = talib.ATR(data["High"], data["Low"], data["Close"], timeperiod=period)
    return atr


def calculate_macd_histogram(data, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) Histogram.

    :param data: A Pandas Series or DataFrame with the price data.
    :param fastperiod: The short-term EMA period.
    :param slowperiod: The long-term EMA period.
    :param signalperiod: The signal line EMA period.
    :return: A Series with the MACD Histogram.
    """
    macd, signal, hist = talib.MACD(
        data, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
    )
    return hist


def calculate_obv(data):
    """
    Calculate On Balance Volume (OBV).

    :param data: A Pandas DataFrame with the close price and volume data.
    :return: A Series with the OBV.
    """
    obv = talib.OBV(data["Close"], data["Volume"])
    return obv


def calculate_cci(data, period=14):
    """
    Calculate the Commodity Channel Index (CCI).

    :param data: A Pandas DataFrame with the high, low, and close price data.
    :param period: The number of periods to use for the calculation.
    :return: A Series with the CCI.
    """
    cci = talib.CCI(data["High"], data["Low"], data["Close"], timeperiod=period)
    return cci
