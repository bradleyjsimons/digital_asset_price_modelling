"""
This module contains the functionality for training and loading a Deep Q-Network (DQN) model for a trading environment.

The main functions in this module are `train_model` and `load_trained_model`. `train_model` takes in market data, initializes a trading environment with this data, 
loads a DQN model, and runs a simulation to train the model. The simulation runs for a specified number of episodes, 
and in each episode, it runs a loop for a specified number of steps. In each step, it predicts an action based on the current state, 
takes that action in the environment to get the next state and reward, and updates the model based on these results. 
After the training, the model is saved with a timestamp in its filename. `load_trained_model` loads a DQN model from a file.

Functions:
    train_model(data: pandas.DataFrame) -> dqn.DQN: Trains a DQN model using the provided data, saves the trained model, and returns the model.
    load_trained_model(model_path: str) -> keras.Model: Loads a DQN model from a file and returns the model.

Constants:
    NUM_EPISODES (int): The number of episodes to run in the simulation.
    MAX_STEPS (int): The maximum number of steps to run in each episode.
    BATCH_SIZE (int): The size of the batch used when updating the model.
"""
import os

from src.learning.rl.environment import TradingEnvironment
from src.learning.rl.models import dqn
from keras.models import load_model


NUM_EPISODES = 20
MAX_STEPS = 20
BATCH_SIZE = 64


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
    action_size = 3  # For example, if the actions are "buy", "sell", and "hold"

    # Load the DQN model
    print("loading DQN model...")
    model = dqn.DQN(state_size, action_size)

    # Run the simulation
    print("running simulation...")
    for episode in range(NUM_EPISODES):
        print(f"Starting episode {episode+1} of {NUM_EPISODES}")
        state = env.reset()

        for step in range(MAX_STEPS):
            print(f"\tStep {step+1} of {MAX_STEPS}")
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
