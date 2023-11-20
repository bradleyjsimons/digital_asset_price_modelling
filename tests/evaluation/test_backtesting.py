"""
This module contains tests for the functions in the backtesting module.

Tests cover the calculation of backtest returns based on the trained model and market data.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock
from src.evaluation.backtesting import (
    calculate_backtest_returns,
    calculate_benchmark_returns,
)


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df = pd.DataFrame(
        {
            "lstm_feature": [0.01, 0.02, -0.01, 0.03, -0.02],
            "target": [1, 0, 1, 0, 1],
            "log_return": [0.02, 0.03, -0.01, 0.04, -0.01],
        }
    )
    return df


@pytest.fixture
def mock_model():
    """
    A pytest fixture that creates a mock model for testing.
    """
    model = Mock()
    model.predict.return_value = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.2, 0.5, 0.3],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
        ]
    )
    return model


@pytest.fixture
def mock_scaler():
    """
    A pytest fixture that creates a mock scaler for testing.
    """
    scaler = Mock()
    scaler.inverse_transform.side_effect = lambda x: np.ones(x.shape)
    return scaler


def test_calculate_backtest_returns(mock_model, mock_data, mock_scaler):
    """
    Test the calculate_backtest_returns function.
    """
    # Mock the inverse_transform method to return the correct shape
    mock_scaler.inverse_transform.return_value = np.array(
        [0.01, 0.02, -0.01, 0.03, -0.02]
    ).reshape(-1, 1)

    return_df = calculate_backtest_returns(mock_model, mock_data, mock_scaler)

    assert isinstance(return_df, pd.DataFrame)
    assert set(return_df.columns) == set(
        [
            "strategy_return",
            "cumulative_strategy_return",
        ]
    )
    assert not return_df.isnull().values.any()


def test_calculate_benchmark_returns(mock_data):
    """
    Test the calculate_benchmark_returns function.
    """
    benchmark_df = calculate_benchmark_returns(mock_data)

    assert isinstance(benchmark_df, pd.DataFrame)
    assert set(benchmark_df.columns) == set(
        [
            "benchmark_return_step",
            "cumulative_benchmark_return",
        ]
    )
    assert not benchmark_df.isnull().values.any()
