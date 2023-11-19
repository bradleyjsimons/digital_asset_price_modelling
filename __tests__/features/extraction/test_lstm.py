import numpy as np
import pandas as pd
import pytest
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, TimeDistributed
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
    assert result.shape == (90, 10, 25)  # Check the shape of the output


def test_build_lstm_model():
    """
    Test the build_lstm_model function from the lstm module.
    """
    model = lstm.build_lstm_model((10, 25))
    assert isinstance(model, Sequential)
    assert len(model.layers) == 5  # Check the number of layers in the model
    assert isinstance(model.layers[0], LSTM)
    assert isinstance(model.layers[1], Dropout)
    assert isinstance(model.layers[2], LSTM)
    assert isinstance(model.layers[3], Dropout)
    assert isinstance(model.layers[4], TimeDistributed)


def test_train_model(mocker, mock_data):
    """
    Test the train_model function from the lstm module.
    """
    mock_model = mocker.Mock(spec=Sequential)
    mock_model.fit.return_value = None

    X = lstm.create_sequences(mock_data, sequence_length=10)

    lstm.train_model(mock_model, X)

    # Check that the fit method was called with the correct arguments
    mock_model.fit.assert_called_once_with(X, X, epochs=10, batch_size=32, verbose=1)


def test_extract_features(mocker, mock_data):
    """
    Test the extract_features function from the lstm module.
    """
    # Create a real model with LSTM and Dense layers
    real_model = Sequential()
    real_model.add(
        LSTM(50, activation="relu", input_shape=(10, 25), return_sequences=True)
    )
    real_model.add(Dropout(0.1))
    real_model.add(LSTM(50, activation="relu", return_sequences=True))
    real_model.add(Dropout(0.1))
    real_model.add(TimeDistributed(Dense(25)))

    sequences = lstm.create_sequences(mock_data, sequence_length=10)

    result = lstm.extract_features(real_model, sequences)

    assert isinstance(result, np.ndarray)
    assert result.shape == (90, 10, 50)  # Check the shape of the output
