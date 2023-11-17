"""
This module contains the implementation of a Deep Q-Network (DQN) model for a trading environment.

The DQN model is a type of artificial neural network used as a function approximator in Q-learning, 
a type of model-free reinforcement learning algorithm. The DQN model uses experience replay and a target network 
to stabilize the learning process.

The main class in this module is `DQN`, which provides methods for building the model, predicting actions based on the current state, 
updating the model based on the results of taking an action, and storing experiences in memory.

Classes:
    DQN: Represents a DQN model.

"""
import numpy as np
import tensorflow as tf
from collections import deque
import random
from keras.models import load_model


class DQN:
    def __init__(self, state_size, action_size):
        """
        Initializes the DQN model.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """
        Builds the DQN model.

        Returns:
            model: The DQN model.
        """
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation="relu")
        )
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(32, activation="relu"))
        model.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def update_target_model(self):
        """
        Updates the target model weights with the current model weights.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Returns the action to take based on the current state.

        Args:
            state (numpy.array): The current state.

        Returns:
            action (int): The action to take.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)  # Add an extra dimension for batch size
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        """
        Trains the model using randomly selected experiences in the replay memory.

        Args:
            batch_size (int): The size of the batch of experiences to use for training.
        """
        minibatch = random.sample(self.memory, batch_size)

        # Decay epsilon after each replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape the next_state
                next_state = np.reshape(next_state, [1, self.state_size])
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            # Reshape the state
            state = np.reshape(state, [1, self.state_size])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, model_path):
        """
        Saves the current model to a file.

        Args:
            model_path (str): The path to the file where the model should be saved.
        """
        print("saving model")
        self.model.save(model_path)

    def load_model(self, model_path):
        """
        Loads a model from a file into self.model.

        Args:
            model_path (str): The path to the saved model file.
        """
        print("loading model")
        self.model = load_model(model_path)
