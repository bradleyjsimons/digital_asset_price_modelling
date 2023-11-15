"""
This module provides a main function to control the data fetching, cleaning, and feature engineering process.

Function:
- main: Fetches Bitcoin data, adds blockchain data, adds technical indicators, cleans the data, and extracts features.

This module uses functions from the src.api, src.data, and src.features modules.
"""

from sklearn.model_selection import train_test_split


from src.api.yfinance import fetch_bitcoin_data
from src.data.data_cleaning import clean_data
from src.features.feature_engineering import (
    add_all_technical_indicators,
    add_blockchain_data,
    extract_lstm_features,
)


def main(start_date, end_date):
    """
    Main function to control the data fetching, cleaning, and feature engineering process.

    This function fetches Bitcoin data, adds blockchain data, adds technical indicators, cleans the data,
    and finally extracts features using a transformer model.

    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format.
    :return: A numpy array with the extracted features.
    """
    # Fetch initial data
    df = fetch_bitcoin_data(start_date, end_date)

    # Add features
    df = add_all_technical_indicators(df)
    df = add_blockchain_data(df, timespan="3years", start=start_date)

    # add the target variable
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Clean data
    df = clean_data(df)

    # Separate features and target variable
    X = df.drop("target", axis=1)
    y = df["target"]

    # Extract features
    df = extract_lstm_features(df, sequence_length=30, X=X, y=y)

    # Clean data
    df = clean_data(df)

    return df
