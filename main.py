"""
This is the main module for running the trading bot.

This module contains the main function which controls the overall process of training or loading a model, and potentially evaluating it. 
The process is controlled by the `train` variable. If `train` is True, a new model directory is created, data is fetched and preprocessed, 
and a model is trained with this data. If `train` is False, an existing model is loaded from a specified directory.

Functions:
- main: Controls the overall process of training or loading a model, and potentially evaluating it.
"""

import os

from src.data import data_controller
from src.learning import learning_controller
from src.evaluation import evaluation_controller
from src.utils import folder_manager


def main():
    """
    Main function to control the overall process of training or loading a model, and potentially evaluating it.

    The process is controlled by the `train` variable. If `train` is True, a new model directory is created, data is fetched and preprocessed,
    and a model is trained with this data. If `train` is False, an existing model is loaded from a specified directory.

    The start and end dates for the data are defined, and the base model directory is specified.
    If a new model is being trained, a new model directory is created within the base model directory.
    If an existing model is being loaded, the folder name for the specific model is appended to the base model directory.

    After the model is trained or loaded, it can be evaluated (this is currently commented out).
    """
    # Define date parameters
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    should_train = False  # should train new model or not

    model_dir = "src/models/"

    if should_train:
        # create the directory for storing model files
        model_dir = folder_manager.create_model_directory()

        # Fetch and prep the data
        data = data_controller.main(start_date, end_date, model_dir)

        # Train and store model
        model = learning_controller.train_model(data, model_dir)

    else:
        existing_model_folder_name = "20231116"
        model_dir = os.path.join(model_dir, existing_model_folder_name)

        # load the data
        data = data_controller.load_data(os.path.join(model_dir, "data.csv"))

        # load the scaler
        scaler = data_controller.load_scaler(os.path.join(model_dir, "scaler.pkl"))

        # load the model
        model = learning_controller.load_trained_model(
            os.path.join(model_dir, "dqn_model.h5")
        )

    # Evaluate the model
    evaluation_controller.evaluate_model(
        model,
        data,
        scaler,
    )


if __name__ == "__main__":
    main()
