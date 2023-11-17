"""
This module contains the implementation of a trading environment for reinforcement learning.

The trading environment is a custom environment that simulates a financial market, where an agent can take actions to buy, sell, or hold stocks. The state of the environment is defined by the market data, and the reward is calculated based on the change in price resulting from the agent's actions.

Classes:
    TradingEnvironment: Represents the trading environment.
"""
import numpy as np


class TradingEnvironment:
    def __init__(self, data):
        """
        Initializes the trading environment.

        Args:
            data (pandas.DataFrame): The market data.
        """
        self.data = data
        self.reward = 0
        self.done = False
        self.current_step = 0
        self.position = 0

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            state: The initial state.
        """
        self.reward = 0
        self.done = False
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        """
        Takes an action in the environment.

        Args:
            action: The action to take.

        Returns:
            next_state: The next state.
            reward: The reward for taking the action.
            done: Whether the episode is done.
        """
        # Store the 'Close' price at the current step
        current_close = self.data.iloc[self.current_step]["Close"]

        # Update position based on the action
        if action == "buy" and self.position == 0:
            self.position = 1
        elif action == "sell" and self.position == 1:
            self.position = 0

        # Update current step
        self.current_step += 1
        self.current_step = min(self.current_step, len(self.data) - 1)

        # Calculate reward
        if self.position == 1:
            # If holding a position, reward is the change in price
            next_close = self.data.loc[self.current_step, "Close"]
            self.reward = next_close - current_close
        else:
            # If not holding a position, reward is 0
            self.reward = 0

        # Check if the episode is done
        self.done = self.current_step >= len(self.data) - 1

        return self._get_state(), self.reward, self.done

    def _get_state(self):
        """
        Returns the current state.

        Returns:
            state: The current state.
        """
        return self.data.iloc[self.current_step, :]

    def render(self):
        """
        Renders the environment. This is optional and can be used for visualization.
        """
        pass
