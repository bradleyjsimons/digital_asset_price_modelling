"""
This module provides functions for fetching, cleaning, and preprocessing Bitcoin data.

Functions:
- fetch_bitcoin_data: Fetches historical Bitcoin data from Yahoo Finance.
- clean_data: Cleans the data by handling missing values, checking for duplicates, and normalizing.
- add_all_technical_indicators: Adds technical indicators to the data.

This module uses the yfinance library to fetch data, pandas for data manipulation, 
and sklearn.preprocessing for data normalization. It also uses functions from the 
features.ta module to calculate technical indicators.
"""

import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.features.blockchain import (
    get_hash_rate_over_time,
    get_avg_block_size,
    get_network_difficulty,
    get_miners_revenue,
    get_mempool_size,
)

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


def fetch_bitcoin_data(start_date, end_date, interval="1d"):
    """
    Fetches historical Bitcoin (BTC-USD) data from Yahoo Finance.

    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format.
    :param interval: The interval for the data (e.g., '1d' for daily data, '1wk' for weekly data, '1mo' for monthly data).
    :return: A Pandas DataFrame with the historical Bitcoin data.
    """
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(start=start_date, end=end_date, interval=interval)

    # Drop the 'Dividends' and 'Stock Splits' columns
    btc_data = btc_data.drop(columns=["Dividends", "Stock Splits"])

    return btc_data


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


def add_all_technical_indicators(data):
    """
    Add all technical indicators to the data.

    :param data: A Pandas DataFrame with the price data.
    :return: The DataFrame with the added technical indicators. Rows containing NaN values
             due to the calculation of technical indicators are dropped.
    """
    # add the indicators
    data["upper_bb"], data["middle_bb"], data["lower_bb"] = calculate_bollinger_bands(
        data["Close"]
    )
    data["slowk"], data["slowd"] = calculate_stochastic_oscillator(data)
    data["macd"], data["macdsignal"], data["macdhist"] = calculate_macd(data["Close"])
    data["rsi"] = calculate_rsi(data["Close"])
    data["sma"] = calculate_sma(data["Close"])
    data["ema"] = calculate_ema(data["Close"])
    data["atr"] = calculate_atr(data)
    data["macd_hist"] = calculate_macd_histogram(data["Close"])
    data["obv"] = calculate_obv(data)
    data["cci"] = calculate_cci(data)

    # Drop the NaN values
    data = data.dropna()

    return data


def add_blockchain_data(data, timespan="1year", start=None):
    """
    Fetches data from specified blockchain.com API endpoints and adds it to the DataFrame.

    :param data: A Pandas DataFrame with the price data.
    :param timespan: The timespan for the blockchain data (e.g., '1year' for 1 year).
    :return: The DataFrame with the added blockchain data.
    """
    # Fetch the hash rate over time
    hash_rate_data = get_hash_rate_over_time(timespan, start_date=start)

    # Fetch the average block size over time
    avg_block_size_data = get_avg_block_size(timespan, start_date=start)

    # Fetch the network difficulty over time
    network_difficulty_data = get_network_difficulty(timespan, start_date=start)

    # Fetch the miners revenue over time
    miners_revenue_data = get_miners_revenue(timespan, start_date=start)

    # Fetch the mempool size over time
    mempool_size_data = get_mempool_size(timespan, start_date=start)

    # Convert the dates in the 'data' DataFrame to the same format as the dates in the 'hash_rate_df' DataFrame
    data.index = data.index.date

    # Merge the fetched data with the main DataFrame
    data = pd.merge(data, hash_rate_data, how="left", left_index=True, right_index=True)
    data = pd.merge(
        data, avg_block_size_data, how="left", left_index=True, right_index=True
    )
    data = pd.merge(
        data, network_difficulty_data, how="left", left_index=True, right_index=True
    )
    data = pd.merge(
        data, miners_revenue_data, how="left", left_index=True, right_index=True
    )

    # Resample to daily frequency
    mempool_size_data = mempool_size_data.resample("D").mean()

    # Then merge
    data = pd.merge(
        data, mempool_size_data, how="left", left_index=True, right_index=True
    )

    return data
