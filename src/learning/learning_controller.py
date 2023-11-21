"""
This module contains the functionality for preparing data, training, and loading a Deep Q-Network (DQN) model for a trading environment.

The main functions in this module are `prep_data_and_train_model`, `load_model_and_data`, `train_model` and `load_trained_model`. 

`prep_data_and_train_model` fetches and prepares the market data, initializes a trading environment with this data, 
loads a DQN model, and runs a simulation to train the model. 

`load_model_and_data` loads the model, data, and scaler from the specified directory.

`train_model` takes in market data, initializes a trading environment with this data, 
loads a DQN model, and runs a simulation to train the model. 

`load_trained_model` loads a DQN model from a file.

Functions:
    prep_data_and_train_model(start_date: str, end_date: str, base_model_dir: str) -> Tuple[keras.Model, pandas.DataFrame, sklearn.preprocessing.StandardScaler]: Fetches and prepares the data, trains a DQN model using the provided data, saves the trained model, and returns the model, data, and scaler.
    load_model_and_data(model_dir: str, existing_model_folder_name: str) -> Tuple[keras.Model, pandas.DataFrame, sklearn.preprocessing.StandardScaler]: Loads a DQN model, data, and scaler from the specified directory and returns them.
    train_model(data: pandas.DataFrame, model_dir: str) -> dqn.DQN: Trains a DQN model using the provided data, saves the trained model, and returns the model.
    load_trained_model(model_path: str) -> keras.Model: Loads a DQN model from a file and returns the model.

Constants:
    NUM_EPISODES (int): The number of episodes to run in the simulation.
    MAX_STEPS (int): The maximum number of steps to run in each episode.
    BATCH_SIZE (int): The size of the batch used when updating the model.
"""

import os
from keras.models import load_model

from src.learning.rl.environment import TradingEnvironment
from src.learning.rl.models import dqn
from src.data import data_controller
from src.utils import folder_manager


NUM_EPISODES = 10
MAX_STEPS = 10
BATCH_SIZE = 64


def prep_data_and_train_model(start_date, end_date, base_model_dir="src/models/"):
    """
    Fetches and prepares the data, trains a DQN model using the provided data, saves the trained model, and returns the model, data, and scaler.

    Args:
        start_date (str): The start date for the data.
        end_date (str): The end date for the data.
        base_model_dir (str): The base directory where the new model directory should be created.

    Returns:
        model (keras.Model): The trained DQN model.
        data (pandas.DataFrame): The data used for training.
        scaler (sklearn.preprocessing.StandardScaler): The scaler used for data normalization.
    """
    # create the directory for storing model files
    model_dir = folder_manager.create_model_directory(base_model_dir)

    # Fetch and prep the data
    data, scaler = data_controller.main(start_date, end_date, model_dir)

    # Train and store model
    model = train_model(data, model_dir)

    return model, data, scaler


def load_model_and_data(existing_model_folder_name, base_model_dir="src/models/"):
    """
    Loads a DQN model, data, and scaler from the specified directory and returns them.

    Args:
        base_model_dir (str): The directory where the model, data, and scaler are stored.
        existing_model_folder_name (str): The name of the folder where the model, data, and scaler are stored.

    Returns:
        model (keras.Model): The loaded DQN model.
        data (pandas.DataFrame): The loaded data.
        scaler (sklearn.preprocessing.StandardScaler): The loaded scaler.
    """
    model_dir = os.path.join(base_model_dir, existing_model_folder_name)

    # load the data
    data = data_controller.load_data(os.path.join(model_dir, "data.csv"))

    # load the scaler
    scaler = data_controller.load_scaler(os.path.join(model_dir, "scaler.pkl"))

    # load the model
    model = load_trained_model(os.path.join(model_dir, "dqn_model.h5"))

    return model, data, scaler


def train_model(data, model_dir):
    """
    Trains a DQN model using the provided data and saves the trained model.

    The function initializes a trading environment with the provided data, loads a DQN model,
    and runs a simulation to train the model. The simulation runs for a specified number of episodes,
    and in each episode, it runs a loop for a specified number of steps. In each step, it predicts an action
    based on the current state, takes that action in the environment to get the next state and reward,
    and updates the model based on these results. The trained model is saved with a timestamp in its filename.

    Args:
        data (pandas.DataFrame): The data to use for training. This should be in a format that the TradingEnvironment can use.

    Returns:
        model (dqn.DQN): The trained DQN model.
    """
    # Drop the target variable for RL
    training_data = data.drop(columns=["target"])

    # Initialize the trading environment
    print("setting up RL learning environment...")
    env = TradingEnvironment(training_data)

    # Define the state size and action size
    state_size = len(env._get_state())
    action_size = 3  # actions are "buy", "sell", and "hold"

    # Load the DQN model
    print("loading DQN model...")
    model = dqn.DQN(state_size, action_size)

    # Run the simulation
    print("running simulation...")
    for episode in range(NUM_EPISODES):
        # print(f"Starting episode {episode+1} of {NUM_EPISODES}")
        state = env.reset()

        for step in range(MAX_STEPS):
            # print(f"\tStep {step+1} of {MAX_STEPS}")
            action = model.act(state)
            next_state, reward, done = env.step(action)

            # Store the experience in memory
            model.remember(state, action, reward, next_state, done)

            # Update the model
            if len(model.memory) > BATCH_SIZE:
                model.replay(BATCH_SIZE)

            state = next_state

            if done:
                break

    # Define the filename with .h5 extension
    filename = "dqn_model.h5"

    # Create the full path to save the model
    path = os.path.join(model_dir, filename)

    # Save the model with the timestamp in the filename
    model.save_model(path)
    print("model trained and saved")

    # return the trained model only
    return model.model


def load_trained_model(model_path):
    """
    Load a DQN model from a file.

    :param model_path: The path to the file containing the DQN model.
    :return: The loaded DQN model.
    """
    model = load_model(model_path)
    return model
