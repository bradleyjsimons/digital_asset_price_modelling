"""
This module contains the functionality for backtesting a trained model.

The main function in this module is `backtest_model`, which takes in a trained model and market data, 
calculates the target and predicted output, and calculates the return using a vectorized approach.

Functions:
    backtest_model(model: dqn.DQN, data: pandas.DataFrame) -> pandas.Series: Backtests the model using the provided data and returns the returns series.
"""

import pandas as pd
import numpy as np


def backtest_model(model, data, scaler):
    """
    Backtests a trained model using the provided data.

    The function calculates the target and predicted output, and calculates the return using a vectorized approach.

    Args:
        model (dqn.DQN): The trained DQN model to backtest.
        data (pandas.DataFrame): The data to use for backtesting.

    Returns:
        returns (pandas.DataFrame): The DataFrame with the backtest results.
    """

    # # Remove the target column from the data before making predictions
    data_without_target = data.drop(columns=["target"])

    # Calculate the predicted output
    predicted_output = model.predict(data_without_target)

    # Select the columns to inverse transform
    cols_to_inverse_transform = [
        col for col in data.columns if col not in ["lstm_feature", "target"]
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
    print(predicted_actions)

    # Create a 'position' column based on the predicted actions
    data["position"] = predicted_actions.replace({"buy": 1, "sell": 0, "hold": np.nan})

    # Get the strategy return series
    data["strategy_return"] = data["position"].shift() * data["log_return"]

    # Sum the log returns
    total_log_return = data["strategy_return"].sum()

    # Convert the total log return to a regular return
    total_regular_return = np.exp(total_log_return) - 1

    # Calculate the log return of a simple buy-and-hold strategy
    benchmark_log_return = data["log_return"].sum()

    # Convert the benchmark log return to a regular return
    benchmark_regular_return = np.exp(benchmark_log_return) - 1

    print("Strategy Return: ", total_regular_return)
    print("Benchmark Return: ", benchmark_regular_return)
