from src.data import data_controller
from src.learning import learning_controller
from src.evaluation import evaluation_controller
from src.utils import folder_manager

# import tensorflow as tf

# print(tf.__version__)


def main():
    # Define date parameters
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    # create the directory for storing model files
    model_dir = folder_manager.create_model_directory()

    # Fetch and prep the data
    data = data_controller.main(start_date, end_date, model_dir)

    # Train and store model
    learning_controller.train_model(data, model_dir)

    # # uncomment to load existing model
    # model_path = "src/learning/trained_models/dqn_model_2023-11-15_18-45-30.h5"

    # Evaluate the model
    # evaluation_controller.evaluate_model(model_dir, data)


if __name__ == "__main__":
    main()
