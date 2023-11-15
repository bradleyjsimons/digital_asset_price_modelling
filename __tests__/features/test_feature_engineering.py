"""
This module contains tests for the feature_engineering module.

The tests cover the main functions which add blockchain data and technical indicators to a DataFrame, and extract features from the data.

Functions:
- test_add_blockchain_data: Tests the add_blockchain_data function from the feature_engineering module.
- test_add_all_technical_indicators: Tests the add_all_technical_indicators function from the feature_engineering module.
- test_extract_features: Tests the extract_features function from the feature_engineering module.
"""

import pandas as pd
import pytest
from src.features import feature_engineering as fe


@pytest.fixture
def mock_data(mocker):
    """
    A pytest fixture that mocks the data returned by the blockchain and ta modules.
    """
    df = pd.DataFrame(
        {
            "Close": [1, 2, 3],
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Volume": [1, 2, 3],
        },
        index=pd.date_range(start="1/1/2022", periods=3),
    )

    hash_rate_df = pd.DataFrame(
        {"HashRate": [1, 2, 3]}, index=pd.date_range(start="1/1/2022", periods=3)
    )
    avg_block_size_df = pd.DataFrame(
        {"AvgBlockSize": [1, 2, 3]}, index=pd.date_range(start="1/1/2022", periods=3)
    )
    network_difficulty_df = pd.DataFrame(
        {"NetworkDifficulty": [1, 2, 3]},
        index=pd.date_range(start="1/1/2022", periods=3),
    )
    miners_revenue_df = pd.DataFrame(
        {"MinersRevenue": [1, 2, 3]}, index=pd.date_range(start="1/1/2022", periods=3)
    )
    mempool_size_df = pd.DataFrame(
        {"MempoolSize": [1, 2, 3]}, index=pd.date_range(start="1/1/2022", periods=3)
    )

    mocker.patch(
        "src.features.feature_engineering.get_hash_rate_over_time",
        return_value=hash_rate_df,
    )
    mocker.patch(
        "src.features.feature_engineering.get_avg_block_size",
        return_value=avg_block_size_df,
    )
    mocker.patch(
        "src.features.feature_engineering.get_network_difficulty",
        return_value=network_difficulty_df,
    )
    mocker.patch(
        "src.features.feature_engineering.get_miners_revenue",
        return_value=miners_revenue_df,
    )
    mocker.patch(
        "src.features.feature_engineering.get_mempool_size",
        return_value=mempool_size_df,
    )

    mocker.patch(
        "src.features.ta.calculate_bollinger_bands",
        return_value=(df["Close"], df["Close"], df["Close"]),
    )
    mocker.patch(
        "src.features.ta.calculate_stochastic_oscillator",
        return_value=(df["Close"], df["Close"]),
    )
    mocker.patch(
        "src.features.ta.calculate_macd",
        return_value=(df["Close"], df["Close"], df["Close"]),
    )
    mocker.patch("src.features.ta.calculate_rsi", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_sma", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_ema", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_atr", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_macd_histogram", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_obv", return_value=df["Close"])
    mocker.patch("src.features.ta.calculate_cci", return_value=df["Close"])
    return df


def test_add_blockchain_data(mock_data):
    """
    Test the add_blockchain_data function from the feature_engineering module.
    """
    result = fe.add_blockchain_data(mock_data)
    assert isinstance(result, pd.DataFrame)


def test_add_all_technical_indicators(mock_data):
    """
    Test the add_all_technical_indicators function from the feature_engineering module.
    """
    result = fe.add_all_technical_indicators(mock_data)
    assert isinstance(result, pd.DataFrame)


# def test_extract_features(mocker, mock_data):
#     """
#     Test the extract_features function from the feature_engineering module.
#     """
#     mocker.patch(
#         "src.features.extraction.transformer.create_sequences", return_value=mock_data
#     )
#     mocker.patch(
#         "src.features.extraction.transformer.build_transformer_model", return_value=None
#     )
#     mocker.patch("src.features.extraction.transformer.train_model", return_value=None)
#     mocker.patch(
#         "src.features.extraction.transformer.extract_transformer_features",
#         return_value=mock_data,
#     )
#     result = fe.extract_features(
#         mock_data, sequence_length=30, X_train=mock_data, y_train=mock_data["Close"]
#     )
#     assert isinstance(result, pd.DataFrame)
