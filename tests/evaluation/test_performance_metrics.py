"""
This module contains tests for the functions in the performance_metrics module.

Tests cover the calculation of various performance metrics based on the backtest results.
"""

import pandas as pd
import numpy as np
import pytest
from src.evaluation.performance_metrics import (
    calculate_performance_metrics,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_risk_adjusted_return,
    calculate_volatility,
    calculate_beta,
    calculate_alpha,
)


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df = pd.DataFrame(
        {
            "strategy_return": [0.01, 0.02, -0.01, 0.03, -0.02],
            "benchmark_return_step": [0.02, 0.03, -0.01, 0.04, -0.01],
            "cumulative_strategy_return": [1.01, 1.03, 1.02, 1.05, 1.03],
        }
    )
    return df


def test_calculate_performance_metrics(mock_data):
    """
    Test the calculate_performance_metrics function.
    """
    metrics = calculate_performance_metrics(mock_data)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == set(
        [
            "sharpe_ratio",
            "max_drawdown",
            "risk_adjusted_return",
            "volatility",
            "beta",
            "alpha",
        ]
    )


def test_calculate_sharpe_ratio(mock_data):
    """
    Test the calculate_sharpe_ratio function.
    """
    sharpe_ratio = calculate_sharpe_ratio(mock_data)
    expected_sharpe_ratio = (
        mock_data["strategy_return"].mean() / mock_data["strategy_return"].std()
    )

    assert isinstance(sharpe_ratio, float)
    assert np.isclose(sharpe_ratio, expected_sharpe_ratio)


def test_calculate_max_drawdown(mock_data):
    """
    Test the calculate_max_drawdown function.
    """
    max_drawdown = calculate_max_drawdown(mock_data)
    running_max = np.maximum.accumulate(mock_data["cumulative_strategy_return"])
    drawdowns = 1 - mock_data["cumulative_strategy_return"] / running_max
    expected_max_drawdown = np.max(drawdowns)

    assert isinstance(max_drawdown, float)
    assert np.isclose(max_drawdown, expected_max_drawdown)


def test_calculate_risk_adjusted_return(mock_data):
    """
    Test the calculate_risk_adjusted_return function.
    """
    risk_adjusted_return = calculate_risk_adjusted_return(mock_data)
    expected_risk_adjusted_return = (
        mock_data["strategy_return"].sum() / mock_data["strategy_return"].std()
    )

    assert isinstance(risk_adjusted_return, float)
    assert np.isclose(risk_adjusted_return, expected_risk_adjusted_return)


def test_calculate_volatility(mock_data):
    """
    Test the calculate_volatility function.
    """
    volatility = calculate_volatility(mock_data)
    expected_volatility = mock_data["strategy_return"].std()

    assert isinstance(volatility, float)
    assert np.isclose(volatility, expected_volatility)


def test_calculate_beta(mock_data):
    """
    Test the calculate_beta function.
    """
    beta = calculate_beta(mock_data)
    covariance = (
        mock_data[["strategy_return", "benchmark_return_step"]].cov().iloc[0, 1]
    )
    variance = mock_data["benchmark_return_step"].var()
    expected_beta = covariance / variance

    assert isinstance(beta, float)
    assert np.isclose(beta, expected_beta)


def test_calculate_alpha(mock_data):
    """
    Test the calculate_alpha function.
    """
    alpha = calculate_alpha(mock_data)
    expected_alpha = (
        mock_data["strategy_return"].sum() - mock_data["benchmark_return_step"].sum()
    )

    assert isinstance(alpha, float)
    assert np.isclose(alpha, expected_alpha)
