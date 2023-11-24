"""
This module provides functions to control the data fetching, cleaning, feature engineering process, and loading of data and scalers.

Functions:
- main: Fetches Bitcoin data, adds blockchain data, adds technical indicators, normalizes the data, extracts features using an LSTM model, and stores the resulting DataFrame in a CSV file.
- load_data: Loads a DataFrame from a CSV file.
- load_scaler: Loads a scaler object from a file.

This module uses functions from the src.api, src.data, and src.features modules. The resulting DataFrame is stored in a CSV file in the specified model directory.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

from src.api.yfinance import fetch_bitcoin_data
from src.data.data_cleaning import clean_data, normalize_data
from src.features.feature_engineering import (
    add_all_technical_indicators,
    add_blockchain_data,
    extract_lstm_features,
)


def main(start_date, end_date, model_dir):
    """
    Main function to control the data fetching, cleaning, and feature engineering process.

    This function fetches Bitcoin data, adds blockchain data, adds technical indicators, normalizes the data,
    and finally extracts features using an LSTM model. The resulting DataFrame is stored in a CSV file in the
    specified model directory.

    :param start_date: The start date for the data in YYYY-MM-DD format.
    :param end_date: The end date for the data in YYYY-MM-DD format.
    :param model_dir: The directory where the resulting DataFrame will be stored as a CSV file.
    :return: A cleaned DataFrame with the extracted features and target variable.
    """
    # Fetch initial data
    print("fetching data...")
    df = fetch_bitcoin_data(start_date, end_date)

    # Add features
    print("adding features...")
    df = add_all_technical_indicators(df)
    df = add_blockchain_data(df, timespan="5years", start=start_date)

    # Temporarily remove the log returns column
    log_returns = df.pop("log_return")

    # Normalize the data before extraction
    print("normalizing data...")
    df, scaler = normalize_data(df)

    # Extract features
    print("extracting additional features using lstm...")
    df = extract_lstm_features(df, sequence_length=30)

    # add the target variable
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Clean data before modelling
    print("cleaning data for model...")
    df = clean_data(df)

    # Analyze feature importance
    print("analyzing feature importance...")
    top_features = analyze_feature_importance(df)

    # Remove the target adn lstm features columns before inverse transforming
    target = df.pop("target")
    lstm_features = df.pop("lstm_feature")

    # Inverse transform the data before saving
    df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns, index=df.index)

    # Add the log returns, lstm features, and target columns back to the DataFrame
    df["target"] = target
    df["log_return"] = log_returns
    df["lstm_feature"] = lstm_features

    # Select only the top features and the target column
    df = df[top_features + ["target"] + ["log_return"]]

    # store the data in the model directory
    df.to_csv(f"{model_dir}/data.csv")

    return df


def load_data(data_path):
    """
    Load a DataFrame from a CSV file.

    :param data_path: The path to the CSV file.
    :return: A DataFrame containing the data.
    """
    data = pd.read_csv(data_path, index_col=0)

    # convert the index to datetime
    data.index = pd.to_datetime(data.index)

    # set the frequency of the index
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq="D")

    return data


def load_scaler(scaler_path):
    """
    Load a scaler object from a file.

    :param scaler_path: The path to the file containing the scaler object.
    :return: The loaded scaler object.
    """
    scaler = joblib.load(scaler_path)
    return scaler


def analyze_feature_importance(df, target_column="target", top_percent=0.5):
    """
    Analyze feature importance using a Random Forest model and return the top percent of features.

    :param df: The DataFrame containing the features and target variable.
    :param target_column: The name of the target variable column.
    :param top_percent: The top percent of features to keep based on their importance. Default is 0.5 (50%).
    :return: List of top features.
    """
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    # Create and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame of feature importances
    feature_importances = pd.DataFrame(
        {"feature": features.columns, "importance": importances}
    )

    # Sort the DataFrame by importance
    feature_importances.sort_values("importance", ascending=False, inplace=True)

    # Calculate the number of features to keep
    num_features = int(len(features.columns) * top_percent)

    # Get the top features
    top_features = feature_importances["feature"][:num_features].tolist()

    # Print feature importances
    for i, row in feature_importances.iterrows():
        print(f"{row['feature']}: {row['importance']}")

    return top_features
