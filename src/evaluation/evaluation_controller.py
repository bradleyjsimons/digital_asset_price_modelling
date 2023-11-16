"""
This module contains the functionality for evaluating a trained model.

The main function in this module is `evaluate_model`, which takes in a trained model and market data, 
backtests the model, calculates various performance metrics, and creates visualizations.

Functions:
    evaluate_model(model: dqn.DQN, data: pandas.DataFrame): Evaluates the model using the provided data.
"""
from keras.models import load_model

from src.evaluation import backtesting

# from src.learning import metrics
# from src.learning import visualizations


def evaluate_model(model_path, data):
    """
    Evaluates a trained model using the provided data.

    The function loads the model from a file, backtests the model, calculates various performance metrics, and creates visualizations.

    Args:
        model_path (str): The path to the saved model file.
        data (pandas.DataFrame): The data to use for evaluation.
    """
    # Load the model from the file
    model = load_model(model_path)

    # Backtest the model
    total_profit = backtesting.backtest_model(model, data)

    # # Calculate performance metrics
    # sharpe_ratio = metrics.calculate_sharpe_ratio(total_profit)
    # max_drawdown = metrics.calculate_max_drawdown(total_profit)

    # # Create visualizations
    # visualizations.plot_cumulative_returns(total_profit)
    # visualizations.plot_growth_of_dollar(total_profit)
