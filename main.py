from src.data import data_controller
from src.learning import learning_controller
from src.evaluation import evaluation_controller

# import tensorflow as tf

# print(tf.__version__)


def main():
    # Define date parameters
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    # Fetch and prep the data
    data = data_controller.main(start_date, end_date)

    # # Train and store model
    # model_path = learning_controller.train_model(data)

    # uncomment to load existing model
    model_path = "src/learning/trained_models/dqn_model_2023-11-15_18-45-30.h5"

    # Evaluate the model
    evaluation_controller.evaluate_model(model_path, data)


if __name__ == "__main__":
    main()
