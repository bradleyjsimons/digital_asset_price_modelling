"""
This module contains tests for the train_model and load_trained_model functions in the learning_controller module.

Tests cover the training of a DQN model and loading a trained model from a file.
"""

import os
import pytest
import pandas as pd
from keras.models import Sequential

from src.learning.learning_controller import train_model, load_trained_model, BATCH_SIZE


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df = pd.DataFrame(
        {"Close": [1, 2, 3], "log_return": [0.1, 0.2, 0.3], "target": [0, 1, 0]}
    )
    return df


def test_train_model(mocker, mock_data, tmpdir):
    """
    Test the train_model function from the learning_controller module.
    """
    # Mock the DQN model
    mock_model = mocker.patch("src.learning.learning_controller.dqn.DQN")
    instance = mock_model.return_value
    instance.model = Sequential()
    instance.save_model = mocker.MagicMock()  # Mock the save_model method
    instance.replay = mocker.MagicMock()  # Mock the replay method

    # Set instance.memory to a real list and add dummy data
    instance.memory = [(0, 0, 0, 0, False)] * (BATCH_SIZE + 1)

    # Call the function
    model = train_model(mock_data, tmpdir)

    # Check that the function returns a Sequential model
    assert isinstance(model, Sequential)

    # Check that the save_model method was called with the correct arguments
    model_path = os.path.join(tmpdir, "dqn_model.h5")
    instance.save_model.assert_called_once_with(model_path)

    # Check that the replay method was called
    instance.replay.assert_called()


def test_load_trained_model(mocker, tmpdir):
    """
    Test the load_trained_model function from the learning_controller module.
    """
    # Mock the load_model function to return a Sequential model
    mock_load_model = mocker.patch("src.learning.learning_controller.load_model")
    mock_load_model.return_value = Sequential()

    # Create a dummy model file
    model_path = tmpdir.join("dqn_model.h5")
    model_path.write("dummy content")

    # Call the function
    model = load_trained_model(model_path)

    # Check that the function returns a Sequential model
    assert isinstance(model, Sequential)

    # Check that the load_model function was called with the correct arguments
    mock_load_model.assert_called_once_with(model_path)
