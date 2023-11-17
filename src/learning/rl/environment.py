"""
This module contains the implementation of a trading environment for reinforcement learning.

The trading environment is a custom environment that simulates a financial market, where an agent can take actions to buy, sell, or hold stocks. The state of the environment is defined by the market data, and the reward is calculated based on the change in price resulting from the agent's actions.

Classes:
    TradingEnvironment: Represents the trading environment.
"""
import numpy as np


class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        """
        Initializes the trading environment.

        Args:
            data (pandas.DataFrame): The market data.
            initial_balance (float): The initial balance of the agent.
        """
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
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
        self.balance = self.initial_balance
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
        # Map the action from integer to string
        action_map = {0: "hold", 1: "buy", 2: "sell"}
        action = action_map.get(action, "hold")

        # if action is taken, then apply a trade fee
        if action != "hold":
            fee = self.calculate_fee(self.balance)
            self.balance -= fee

        # Update position based on the action and update balance
        if action == "buy" and self.position == 0:
            self.position = 1
        elif action == "sell" and self.position == 1:
            self.position = 0

        # Calculate reward after position change
        # since the reward uses the log return of this step and
        # the next step
        self.reward = np.exp(self.calculate_reward())

        # Update balance based on the reward
        if self.position == 1:
            self.balance *= self.reward

        # Update current step
        self.current_step += 1
        self.current_step = min(self.current_step, len(self.data) - 1)

        # Check if the episode is done
        self.done = self.current_step >= len(self.data) - 1

        # print the current state
        print(
            f"Action: {action}, Position: {self.position}, Balance: {self.balance}, Reward: {self.reward}, Done: {self.done}, Step Return: {np.exp(self.data.iloc[self.current_step, :]['log_return'])}"
        )

        return self._get_state(), self.reward, self.done

    def calculate_reward(self):
        """
        Calculates the reward based on the pre-calculated log returns.

        Returns:
            reward (float): The calculated reward.
        """
        if self.position == 1:
            # If holding a position, reward is the log return of the next price relative to the current price
            reward = self.data.iloc[self.current_step, :]["log_return"]

        else:
            # If not holding a position, reward is 0
            reward = 0

        return reward

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

    def calculate_fee(self, trade_size):
        """
        Calculates the trading fee based on the trade size.

        Args:
            trade_size (float): The size of the trade.

        Returns:
            fee (float): The trading fee.
        """
        # Define the fee tiers
        fee_tiers = [
            (500000000, 0.0004),
            (250000000, 0.0006),
            (100000000, 0.0008),
            (10000000, 0.0010),
            (5000001, 0.0012),
            (2500001, 0.0014),
            (1000001, 0.0016),
            (500001, 0.0018),
            (250001, 0.0020),
            (100001, 0.0022),
            (50001, 0.0024),
            (0, 0.0026),
        ]

        # Find the correct fee tier
        for volume, fee in fee_tiers:
            if trade_size >= volume:
                return trade_size * fee

        # Default fee if no tier is found (should not happen)
        return trade_size * 0.0026
