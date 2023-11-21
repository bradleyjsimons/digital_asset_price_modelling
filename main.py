"""
This is the main module for running the trading bot.

This module contains the main function which controls the overall process of training or loading a model, and potentially evaluating it. 
The process is controlled by the `train` variable. If `train` is True, a new model directory is created, data is fetched and preprocessed, 
and a model is trained with this data. If `train` is False, an existing model is loaded from a specified directory.

Functions:
- main: Controls the overall process of training or loading a model, and potentially evaluating it.
"""


from src.learning import learning_controller
from src.evaluation import evaluation_controller


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

    should_train = True  # should train new model or not

    if should_train:
        model, data, scaler = learning_controller.prep_data_and_train_model(
            start_date, end_date
        )
    else:
        existing_model_folder_name = "20231120"
        model, data, scaler = learning_controller.load_model_and_data(
            existing_model_folder_name
        )

    # Evaluate the model
    evaluation_controller.evaluate_models(
        [model],
        data,
        scaler,
    )


if __name__ == "__main__":
    main()
