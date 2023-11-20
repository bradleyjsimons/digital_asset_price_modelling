# src/evaluation/performance_metrics.py
"""
This module contains the functionality for calculating performance metrics of a trading strategy.

Functions:
    calculate_performance_metrics(backtest_df: pandas.DataFrame) -> dict: Calculates various performance metrics based on the backtest results.
    calculate_sharpe_ratio(backtest_df: pandas.DataFrame) -> float: Calculates the Sharpe ratio based on the backtest results.
    calculate_max_drawdown(backtest_df: pandas.DataFrame) -> float: Calculates the maximum drawdown based on the backtest results.
    calculate_risk_adjusted_return(backtest_df: pandas.DataFrame) -> float: Calculates the risk-adjusted return based on the backtest results.
    calculate_volatility(backtest_df: pandas.DataFrame) -> float: Calculates the volatility based on the backtest results.
    calculate_beta(backtest_df: pandas.DataFrame) -> float: Calculates the beta based on the backtest results.
    calculate_alpha(backtest_df: pandas.DataFrame) -> float: Calculates the alpha based on the backtest results.
"""

import pandas as pd
import numpy as np


def calculate_performance_metrics(backtest_df):
    """
    Calculates various performance metrics based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        metrics (dict): A dictionary with the calculated performance metrics.
    """
    # Initialize a dictionary to store the metrics
    metrics = {}

    # Calculate performance metrics
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(backtest_df)
    metrics["max_drawdown"] = calculate_max_drawdown(backtest_df)
    metrics["risk_adjusted_return"] = calculate_risk_adjusted_return(backtest_df)
    metrics["volatility"] = calculate_volatility(backtest_df)
    metrics["beta"] = calculate_beta(backtest_df)
    metrics["alpha"] = calculate_alpha(backtest_df)

    return metrics


def calculate_sharpe_ratio(backtest_df):
    """
    Calculates the Sharpe ratio based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        sharpe_ratio (float): The calculated Sharpe ratio.
    """
    # Assume a risk-free rate of 0
    risk_free_rate = 0

    # Calculate the excess returns
    excess_returns = backtest_df["strategy_return"] - risk_free_rate

    # Calculate the Sharpe ratio
    sharpe_ratio = excess_returns.mean() / excess_returns.std()

    return sharpe_ratio


def calculate_max_drawdown(backtest_df):
    """
    Calculates the maximum drawdown based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        max_drawdown (float): The calculated maximum drawdown.
    """
    # Calculate the cumulative returns
    cumulative_returns = backtest_df["cumulative_strategy_return"]

    # Calculate the running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate the drawdowns
    drawdowns = 1 - cumulative_returns / running_max

    # Calculate the maximum drawdown
    max_drawdown = np.max(drawdowns)

    return max_drawdown


def calculate_risk_adjusted_return(backtest_df):
    """
    Calculates the risk-adjusted return based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        risk_adjusted_return (float): The calculated risk-adjusted return.
    """
    # Calculate the return
    total_return = backtest_df["strategy_return"].sum()

    # Calculate the volatility
    volatility = backtest_df["strategy_return"].std()

    # Calculate the risk-adjusted return
    risk_adjusted_return = total_return / volatility

    return risk_adjusted_return


def calculate_volatility(backtest_df):
    """
    Calculates the volatility based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        volatility (float): The calculated volatility.
    """
    # Calculate the volatility
    volatility = backtest_df["strategy_return"].std()

    return volatility


def calculate_beta(backtest_df):
    """
    Calculates the beta based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        beta (float): The calculated beta.
    """
    # Calculate the covariance between the strategy return and the benchmark return
    covariance = (
        backtest_df[["strategy_return", "benchmark_return_step"]].cov().iloc[0, 1]
    )

    # Calculate the variance of the benchmark return
    variance = backtest_df["benchmark_return_step"].var()

    # Calculate the beta
    beta = covariance / variance

    return beta


def calculate_alpha(backtest_df):
    """
    Calculates the alpha based on the backtest results.

    Args:
        backtest_df (pandas.DataFrame): The DataFrame with the backtest results.

    Returns:
        alpha (float): The calculated alpha.
    """
    # Calculate the total strategy return
    total_strategy_return = backtest_df["strategy_return"].sum()

    # Calculate the total benchmark return
    total_benchmark_return = backtest_df["benchmark_return_step"].sum()

    # Calculate the alpha
    alpha = total_strategy_return - total_benchmark_return

    return alpha
