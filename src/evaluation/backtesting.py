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
    # Remove the target column from the data before making predictions
    data_without_target = data.drop(columns=["target"])

    # Calculate the predicted output
    predicted_output = model.predict(data_without_target)

    # Inverse transform the predicted output
    predicted_output = scaler.inverse_transform(predicted_output)

    # Convert the predicted output to a DataFrame
    predicted_output_df = pd.DataFrame(
        predicted_output,
        columns=["buy", "sell", "hold"],
        index=data_without_target.index,
    )

    # Choose the action with the highest probability
    predicted_actions = predicted_output_df.idxmax(axis=1)

    # Create a 'position' column based on the predicted actions
    data["position"] = (
        predicted_actions.replace({"buy": 1, "sell": 0, "hold": np.nan})
        .ffill()
        .fillna(0)
    )

    # fill zero values with a small number
    data.loc[data["Close"] == 0, "Close"] = 1e-9

    # Calculate the daily returns
    data["daily_return"] = data["Close"].pct_change()
    print(data["daily_return"].describe())

    # Calculate the strategy returns
    data["strategy_return"] = data["position"].shift() * data["daily_return"]

    data.dropna(inplace=True)
    print(data["daily_return"].describe())
    print(data["position"].describe())
    print(data["strategy_return"].describe())
    total_strategy_return = data["strategy_return"].sum()
    print(total_strategy_return)

    return data
