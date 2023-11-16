"""
This module provides a function for fetching historical Bitcoin data from Yahoo Finance, calculating log returns, and cleaning the data.

Function:
- fetch_bitcoin_data: Fetches historical Bitcoin data from Yahoo Finance, calculates log returns, and cleans the data.

This module uses the yfinance library to fetch data and the data_cleaning module to clean the data.
"""

import yfinance as yf
import numpy as np

from src.data.data_cleaning import clean_data


def fetch_bitcoin_data(start_date, end_date, interval="1d"):
    """
    Fetches historical Bitcoin (BTC-USD) data from Yahoo Finance, calculates log returns, and cleans the data.

    This function uses the yfinance library to fetch historical Bitcoin data for a specified date range and interval.
    It drops the 'Dividends' and 'Stock Splits' columns from the fetched data.
    It calculates the log return of the 'Close' price and adds it as a new column 'log_return'.
    It then cleans the data by handling missing values and checking for duplicates.

    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format.
    :param interval: The interval for the data (e.g., '1d' for daily data, '1wk' for weekly data, '1mo' for monthly data).
    :return: A cleaned Pandas DataFrame with the historical Bitcoin data and the calculated log returns.
    """
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(start=start_date, end=end_date, interval=interval)

    # Drop the 'Dividends' and 'Stock Splits' columns
    btc_data = btc_data.drop(columns=["Dividends", "Stock Splits"])

    # Clean the data
    btc_data = clean_data(btc_data)

    # Calculate log returns
    btc_data["log_return"] = np.log(btc_data["Close"] / btc_data["Close"].shift(1))

    # Clean the data again now that log returns are included
    btc_data = clean_data(btc_data)

    return btc_data
