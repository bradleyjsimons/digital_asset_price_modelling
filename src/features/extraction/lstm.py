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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import KFold


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


def train_model(model, X, y, n_splits=5):
    """
    Trains the LSTM model with K-fold cross-validation and returns the best model.

    :param model: The LSTM model to be trained.
    :param X: Data.
    :param y: Labels.
    :param n_splits: Number of folds for cross-validation.
    :return: The best LSTM model.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_model = None
    best_score = float("inf")  # for loss lower is better, for accuracy higher is better

    for train_indices, val_indices in kfold.split(X, y):
        x_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        x_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        model.fit(x_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=1)
        scores = model.evaluate(x_val_fold, y_val_fold, verbose=0)

        # If the score for this fold is better than all previous folds, save this model as the best model
        if scores < best_score:
            best_score = scores
            best_model = model

        print(f"Score: {model.metrics_names[0]} of {scores}")

    return best_model


def extract_features(model, sequences):
    """
    Extracts features from the data using the trained LSTM model.

    :param model: The trained LSTM model.
    :param sequences: The sequences from which to extract features.
    :return: A numpy array with the extracted features.
    """
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return feature_extractor.predict(sequences)
