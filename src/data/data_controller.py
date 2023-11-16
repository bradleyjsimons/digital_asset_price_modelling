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
    print("fetching data...")
    df = fetch_bitcoin_data(start_date, end_date)
    print(df)
    quit()

    # Add features
    print("adding features...")
    df = add_all_technical_indicators(df)
    df = add_blockchain_data(df, timespan="5years", start=start_date)

    # Clean data
    print("cleaing data before extraction")
    df = clean_data(df)

    # Extract features
    print("extracting additional features using lstm...")
    df = extract_lstm_features(df, sequence_length=30)

    # add the target variable
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Separate features and target variable
    X = df.drop("target", axis=1)
    y = df["target"]

    # Clean data
    print("cleaing data for model...")
    df = clean_data(df)

    return df
