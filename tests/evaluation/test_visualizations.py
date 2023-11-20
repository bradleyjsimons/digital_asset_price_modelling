# tests/evaluation/test_visualizations.py

"""
This module contains tests for the functions in the visualizations module.

Tests cover the plotting of cumulative returns of multiple strategies and the benchmark.
"""

import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from src.evaluation.visualizations import plot_cumulative_returns


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df1 = pd.DataFrame(
        {
            "cumulative_strategy_return": np.random.rand(100),
        },
        index=pd.date_range(start="1/1/2020", periods=100),
    )
    df2 = pd.DataFrame(
        {
            "cumulative_benchmark_return": np.random.rand(100),
        },
        index=pd.date_range(start="1/1/2020", periods=100),
    )
    return [df1, df2]


def test_plot_cumulative_returns(mock_data, tmpdir, mocker):
    """
    Test the plot_cumulative_returns function.
    """
    # Mock plt.show to prevent the plot from being displayed during the test
    mocker.patch("matplotlib.pyplot.show")

    # Call the function with the mock data
    plot_cumulative_returns(mock_data, ["Strategy", "Benchmark"])

    # Save the plot to a file
    plot_path = tmpdir.join("plot.png")
    plt.savefig(plot_path)

    # Compare the plot with a reference image (not included in this example)
    # Uncomment the following line if you have a reference image
    # assert compare_images("path_to_reference_image.png", plot_path, tol=0) is None

    # Check that the labels and title are correct
    assert plt.gca().get_xlabel() == "Time"
    assert plt.gca().get_ylabel() == "Cumulative Returns"
    assert plt.gca().get_title() == "Cumulative Returns over Time"
    assert plt.gca().get_legend().get_texts()[0].get_text() == "Strategy"
    assert plt.gca().get_legend().get_texts()[1].get_text() == "Benchmark"
