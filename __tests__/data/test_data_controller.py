"""
This module contains tests for the data_controller module.

The tests cover the main function which controls the data fetching, cleaning, and feature engineering process.

Functions:
- test_main: Tests the main function from the data_controller module.
"""

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler
import joblib


from src.data.data_controller import main, load_data, load_scaler


@pytest.fixture
def mock_functions(mocker):
    """
    A pytest fixture that mocks the functions used in the main function.
    """
    df = pd.DataFrame({"Close": [1, 2, 3], "log_return": [0.1, 0.2, 0.3]})

    mocker.patch("src.data.data_controller.fetch_bitcoin_data", return_value=df)
    mocker.patch("src.data.data_controller.clean_data", return_value=df)
    mocker.patch(
        "src.data.data_controller.add_all_technical_indicators",
        return_value=df,
    )
    mocker.patch(
        "src.data.data_controller.add_blockchain_data",
        return_value=df,
    )
    mocker.patch("src.data.data_controller.extract_lstm_features", return_value=df)


def test_main(mock_functions, tmpdir):
    """
    Test the main function from the data_controller module.

    This function tests the main function by checking that it runs without errors and returns a DataFrame.
    """
    model_dir = tmpdir.mkdir("model_dir")
    result, scaler = main("2022-01-01", "2022-01-31", model_dir)

    # Check that the function returns a DataFrame and a scaler
    assert isinstance(result, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)

    # Check that the DataFrame and scaler are saved to the model directory
    assert (model_dir / "data.csv").check()
    assert (model_dir / "scaler.pkl").check()


def test_load_data(tmpdir):
    """
    Test the load_data function from the data_controller module.
    """
    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame({"Close": [1, 2, 3], "log_return": [0.1, 0.2, 0.3]})
    df.index = pd.date_range(start="2022-01-01", periods=len(df), freq="D")
    data_path = tmpdir.join("data.csv")
    df.to_csv(data_path)

    # Load the data using the function
    loaded_data = load_data(data_path)

    # Check that the loaded data is a DataFrame and equals the original DataFrame
    assert isinstance(loaded_data, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded_data, df)


def test_load_scaler(tmpdir):
    """
    Test the load_scaler function from the data_controller module.
    """
    # Create a scaler, fit it with some data, and save it as a file
    scaler = MinMaxScaler()
    data = [[1], [2], [3]]  # Some dummy data
    scaler.fit(data)
    scaler_path = str(tmpdir.join("scaler.pkl"))
    joblib.dump(scaler, scaler_path)

    # Load the scaler using the function
    loaded_scaler = load_scaler(scaler_path)

    # Check that the loaded scaler is a MinMaxScaler and equals the original scaler
    assert isinstance(loaded_scaler, MinMaxScaler)

    assert all(loaded_scaler.min_ == scaler.min_)
    assert all(loaded_scaler.scale_ == scaler.scale_)
