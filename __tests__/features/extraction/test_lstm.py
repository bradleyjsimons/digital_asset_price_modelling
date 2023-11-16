import numpy as np
import pandas as pd
import pytest
from keras.models import Sequential
from keras.layers import Dense, LSTM
from src.features.extraction import lstm


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df = pd.DataFrame(
        np.random.rand(100, 25), columns=[f"feature_{i}" for i in range(25)]
    )
    return df


def test_create_sequences(mock_data):
    """
    Test the create_sequences function from the lstm module.
    """
    result = lstm.create_sequences(mock_data, sequence_length=10)
    assert isinstance(result, np.ndarray)


def test_build_lstm_model():
    """
    Test the build_lstm_model function from the lstm module.
    """
    result = lstm.build_lstm_model()
    assert isinstance(result, Sequential)


def test_train_model(mocker, mock_data):
    """
    Test the train_model function from the lstm module.
    """
    mock_model = mocker.Mock()
    mock_model.fit.return_value = None
    mock_model.evaluate.return_value = 0.5
    mock_model.metrics_names = ["loss"]

    X = mock_data.values
    y = np.random.rand(100)

    result = lstm.train_model(mock_model, X, y)
    assert result is mock_model


def test_extract_features(mocker, mock_data):
    """
    Test the extract_features function from the lstm module.
    """
    # Create a real model with an LSTM layer and a Dense layer
    real_model = Sequential()
    real_model.add(LSTM(10, input_shape=(10, 25)))
    real_model.add(Dense(1))

    # Mock the methods used in the extract_features function
    real_model.predict = mocker.Mock(return_value=np.random.rand(100, 1))

    sequences = lstm.create_sequences(mock_data, sequence_length=10)

    result = lstm.extract_features(real_model, sequences)
    assert isinstance(result, np.ndarray)
