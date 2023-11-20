# src/evaluation/visualizations.py

"""
This module contains the functionality for visualizing the results of backtesting.

The main function in this module is `plot_cumulative_returns`, which takes in a list of DataFrames containing backtest results and a list of labels, and plots the cumulative returns of each strategy and the benchmark.

Functions:
    plot_cumulative_returns(backtest_dfs: list, labels: list): Plots the cumulative returns of multiple strategies and the benchmark.
"""

import matplotlib.pyplot as plt


def plot_cumulative_returns(backtest_dfs, labels):
    """
    Plots the cumulative returns of multiple strategies and the benchmark.

    This function takes in a list of DataFrames, each containing the backtest results of a strategy, and a list of labels for the strategies. It creates a line plot of the cumulative returns over time for each strategy and the benchmark.

    Args:
        backtest_dfs (list of pandas.DataFrame): List of DataFrames, each containing the backtest results of a strategy. Each DataFrame must have a 'cumulative_strategy_return' or 'cumulative_benchmark_return' column and an index representing time.
        labels (list of str): List of labels for the strategies. Each label corresponds to a DataFrame in `backtest_dfs`.

    Returns:
        None. The function creates a plot using matplotlib and displays it using plt.show().
    """
    plt.figure(figsize=(12, 6))

    for backtest_df, label in zip(backtest_dfs, labels):
        if "cumulative_strategy_return" in backtest_df.columns:
            plt.plot(
                backtest_df.index,
                backtest_df["cumulative_strategy_return"],
                label=label,
            )
        elif "cumulative_benchmark_return" in backtest_df.columns:
            plt.plot(
                backtest_df.index,
                backtest_df["cumulative_benchmark_return"],
                label=label,
            )

    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")
    plt.title("Cumulative Returns over Time")
    plt.legend()
    plt.show()
