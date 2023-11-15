"""
This module provides a function for fetching historical Bitcoin data from Yahoo Finance.

Function:
- fetch_bitcoin_data: Fetches historical Bitcoin data from Yahoo Finance.

This module uses the yfinance library to fetch data.
"""

import yfinance as yf


def fetch_bitcoin_data(start_date, end_date, interval="1d"):
    """
    Fetches historical Bitcoin (BTC-USD) data from Yahoo Finance.

    This function uses the yfinance library to fetch historical Bitcoin data for a specified date range and interval.
    It returns a Pandas DataFrame with the fetched data, after dropping the 'Dividends' and 'Stock Splits' columns.

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
