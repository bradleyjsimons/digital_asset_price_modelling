"""
This module contains the functionality for evaluating a trained model.

The main function in this module is `evaluate_model`, which takes in a trained model, a scaler, and market data, 
backtests the model, calculates various performance metrics, and creates visualizations.

Functions:
    evaluate_model(model: keras.Model, scaler: Scaler, data: pandas.DataFrame): Evaluates the model using the provided data.
"""

from src.evaluation import backtesting
from src.evaluation import performance_metrics

# from src.learning import metrics
# from src.learning import visualizations


def evaluate_model(model, data, scaler):
    """
    Evaluates a trained model using the provided data.

    The function backtests the model, calculates various performance metrics, and creates visualizations.

    Args:
        data (pandas.DataFrame): The data to use for evaluation.
        scaler (Scaler): The scaler used for data normalization.
        model (keras.Model): The trained model for evaluation.
    """

    # Backtest the model
    backtest_df = backtesting.calculate_backtest_returns(model, data, scaler)

    # Calculate performance metrics
    metrics = performance_metrics.calculate_performance_metrics(backtest_df)

    import pprint

    pprint.pprint(metrics)

    # # Create visualizations
    # visualizations.plot_cumulative_returns(tot
    # al_profit)
    # visualizations.plot_growth_of_dollar(total_profit)
