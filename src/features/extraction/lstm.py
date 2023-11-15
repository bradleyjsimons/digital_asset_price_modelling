"""
lstm.py

This module provides functions for creating sequences from a DataFrame, building an LSTM model, training the model, and
extracting features from the data using the trained model.

Functions:
- create_sequences: Creates sequences from a DataFrame.
- build_lstm_model: Builds an LSTM model.
- train_model: Trains the LSTM model.
- extract_features: Extracts features from the data using the trained model.
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


def create_sequences(df, sequence_length):
    """
    Creates sequences from a DataFrame.

    :param df: A Pandas DataFrame with the processed data.
    :param sequence_length: The length of the sequences to be created from the data.
    :return: A numpy array with the created sequences.
    """
    features = df.values
    sequences = []
    for i in range(len(features) - sequence_length):
        seq = features[i : i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)


def build_lstm_model():
    """
    Builds an LSTM model.

    :return: The built LSTM model.
    """
    model = Sequential()
    model.add(
        LSTM(50, activation="relu", input_shape=(None, 25), return_sequences=True)
    )
    model.add(Dropout(0.1))
    model.add(LSTM(50, activation="relu"))
    model.add(Dropout(0.1))
    model.add(
        Dense(1, activation="linear")
    )  # Adjust the output layer depending on your requirement
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train):
    """
    Trains the LSTM model.

    :param model: The LSTM model to be trained.
    :param X_train: Training data.
    :param y_train: Training labels.
    :return: The trained LSTM model.
    """
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model


def extract_features(model, sequences):
    """
    Extracts features from the data using the trained LSTM model.

    :param model: The trained LSTM model.
    :param sequences: The sequences from which to extract features.
    :return: A numpy array with the extracted features.
    """
    return model.predict(sequences)
