"""
This module contains the functionality for backtesting a trained model.

The main functions in this module are `calculate_backtest_returns` and `calculate_benchmark_returns`.

`calculate_backtest_returns` takes in a trained model and market data, calculates the target and predicted output, 
and calculates the return using a vectorized approach. It returns a DataFrame with the strategy return and its cumulative value at each time step.

`calculate_benchmark_returns` takes in market data and calculates the benchmark return using a vectorized approach. 
It returns a DataFrame with the benchmark return and its cumulative value at each time step.

Functions:
    calculate_backtest_returns(model: dqn.DQN, data: pandas.DataFrame, scaler: sklearn.preprocessing.StandardScaler) -> pandas.DataFrame: 
        Calculates the strategy returns at each step and their cumulative values, and returns a DataFrame with these series.
        
    calculate_benchmark_returns(data: pandas.DataFrame) -> pandas.DataFrame:
        Calculates the benchmark returns at each step and their cumulative values, and returns a DataFrame with these series.
"""

import pandas as pd
import numpy as np


# src/evaluation/backtesting.py


def calculate_backtest_returns(model, data, scaler):
    """
    Calculates the strategy returns at each step and their cumulative values.

    The function takes in a trained model and market data, calculates the target and predicted output,
    and calculates the return using a vectorized approach.

    Args:
        model (dqn.DQN): The trained DQN model to backtest.
        data (pandas.DataFrame): The data to use for backtesting.
        scaler (sklearn.preprocessing.StandardScaler): The scaler used to inverse transform the data.

    Returns:
        return_df (pandas.DataFrame): A DataFrame with the strategy return and its cumulative value at each time step.
    """

    # Remove the target column from the data before making predictions
    data_without_target = data.drop(columns=["target"])

    # Calculate the predicted output
    predicted_output = model.predict(data_without_target)

    # Select the columns to inverse transform
    cols_to_inverse_transform = [
        col
        for col in data.columns
        if col not in ["lstm_feature", "target", "log_return"]
    ]

    # Inverse transform the selected columns
    data[cols_to_inverse_transform] = scaler.inverse_transform(
        data[cols_to_inverse_transform]
    )

    # Convert the predicted output to a DataFrame
    predicted_output_df = pd.DataFrame(
        predicted_output,
        columns=["buy", "sell", "hold"],
        index=data.index,
    )

    # Choose the action with the highest probability
    predicted_actions = predicted_output_df.idxmax(axis=1)

    # Create a 'position' column based on the predicted actions
    data["position"] = predicted_actions.replace({"buy": 1, "sell": 0, "hold": np.nan})

    # Calculate the strategy return at each step
    data["strategy_return"] = data["position"].shift() * data["log_return"]

    # Calculate the cumulative strategy return at each time step
    data["cumulative_strategy_return"] = np.exp(data["strategy_return"].cumsum()) - 1

    # Create a DataFrame with the strategy return and its cumulative value at each time step
    return_df = data[
        [
            "strategy_return",
            "cumulative_strategy_return",
        ]
    ]

    # Drop rows with NaN values
    return_df = return_df.dropna()

    # Return the DataFrame
    return return_df


def calculate_benchmark_returns(data):
    """
    Calculates the benchmark returns.

    The function takes in market data and calculates the benchmark return using a vectorized approach.

    Args:
        data (pandas.DataFrame): The data to use for calculating benchmark returns.

    Returns:
        benchmark_df (pandas.DataFrame): A DataFrame with the benchmark return and its cumulative value at each time step.
    """
    data = data.copy()  # Create a copy of the input DataFrame

    # Calculate the benchmark return at each step
    data["benchmark_return_step"] = data["log_return"]

    # Calculate the cumulative benchmark return at each time step
    data["cumulative_benchmark_return"] = np.exp(data["log_return"].cumsum()) - 1

    # Create a DataFrame with the benchmark return and its cumulative value at each time step
    benchmark_df = data[
        [
            "benchmark_return_step",
            "cumulative_benchmark_return",
        ]
    ]

    # Drop rows with NaN values
    benchmark_df = benchmark_df.dropna()

    # Return the DataFrame
    return benchmark_df
