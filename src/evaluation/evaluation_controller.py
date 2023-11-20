"""
This module contains the functionality for evaluating a trained model.

The main function in this module is `evaluate_model`, which takes in a trained model, a scaler, and market data, 
backtests the model, calculates various performance metrics, and creates visualizations.

Functions:
    evaluate_model(model: keras.Model, scaler: Scaler, data: pandas.DataFrame): Evaluates the model using the provided data.
"""
import matplotlib.pyplot as plt

from src.evaluation import backtesting
from src.evaluation import performance_metrics
from src.evaluation import visualizations


def evaluate_models(models, data, scaler):
    """
    Evaluates multiple trained models using the provided data.

    The function backtests each model, calculates various performance metrics, and creates visualizations.

    Args:
        models (list of keras.Model): The trained models for evaluation.
        data (pandas.DataFrame): The data to use for evaluation.
        scaler (Scaler): The scaler used for data normalization.
    """
    backtest_dfs = []
    metrics = []

    for model in models:
        # Backtest the model
        backtest_df = backtesting.calculate_backtest_returns(model, data, scaler)
        backtest_dfs.append(backtest_df)

        # Calculate performance metrics
        metric = performance_metrics.calculate_performance_metrics(backtest_df)
        metrics.append(metric)

    # Calculate benchmark returns
    benchmark_df = backtesting.calculate_benchmark_returns(data)
    backtest_dfs.append(benchmark_df)
    labels = ["Model {}".format(i + 1) for i in range(len(models))] + ["Benchmark"]

    # Create visualizations
    visualizations.plot_cumulative_returns(backtest_dfs, labels)
