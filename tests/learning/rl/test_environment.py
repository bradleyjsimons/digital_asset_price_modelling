"""
This module contains tests for the TradingEnvironment class in the environment module.

Tests cover the initialization, reset, step, calculate_reward, _get_state, render, and calculate_fee methods.
"""

import pandas as pd
import numpy as np
import pytest
from src.learning.rl.environment import TradingEnvironment


@pytest.fixture
def mock_data():
    """
    A pytest fixture that creates a mock DataFrame for testing.
    """
    df = pd.DataFrame({"Close": [1, 2, 3], "log_return": [0.1, 0.2, 0.3]})
    return df


def test_trading_environment_initialization(mock_data):
    """
    Test the initialization of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    assert env.data.equals(mock_data)
    assert env.initial_balance == 10000
    assert env.balance == 10000
    assert env.reward == 0
    assert env.done == False
    assert env.current_step == 0
    assert env.position == 0


def test_trading_environment_reset(mock_data):
    """
    Test the reset method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    env.balance = 5000
    env.reward = 1
    env.done = True
    env.current_step = 2
    env.position = 1

    state = env.reset()

    assert env.balance == 10000
    assert env.reward == 0
    assert env.done == False
    assert env.current_step == 0
    assert env.position == 0
    assert state.equals(mock_data.iloc[0, :])


def test_trading_environment_step_sell(mock_data):
    """
    Test the step method of the TradingEnvironment class for a "sell" action.
    """
    env = TradingEnvironment(mock_data)
    env.position = 1  # Set initial position to "buy"
    initial_balance = env.balance
    next_state, reward, done = env.step(2)  # Take "sell" action

    assert env.position == 0
    assert env.balance == initial_balance * np.exp(0.1) * (1 - 0.0026)
    # Balance should decrease due to fee
    assert env.reward == np.exp(0.1)
    assert env.current_step == 1
    assert done == False
    assert next_state.equals(mock_data.iloc[1, :])
    assert reward == env.reward


def test_trading_environment_step_buy(mock_data):
    """
    Test the step method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    initial_balance = env.balance
    next_state, reward, done = env.step(1)

    assert env.position == 1
    assert env.balance == initial_balance * (1 - 0.0026)
    assert env.reward == 0
    assert env.current_step == 1
    assert done == False
    assert next_state.equals(mock_data.iloc[1, :])
    assert reward == env.reward


def test_trading_environment_step_hold(mock_data):
    """
    Test the step method of the TradingEnvironment class for a "hold" action.
    """
    env = TradingEnvironment(mock_data)
    initial_balance = env.balance
    next_state, reward, done = env.step(0)  # Take "hold" action

    assert env.position == 0
    assert env.balance == initial_balance  # Balance should remain the same
    assert env.reward == 0
    assert env.current_step == 1
    assert done == False
    assert next_state.equals(mock_data.iloc[1, :])
    assert reward == env.reward


def test_trading_environment_calculate_reward(mock_data):
    """
    Test the calculate_reward method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    env.position = 1
    env.current_step = 1

    reward = env.calculate_reward()

    assert reward == mock_data.iloc[1, :]["log_return"]


def test_trading_environment_get_state(mock_data):
    """
    Test the _get_state method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    env.current_step = 1

    state = env._get_state()

    assert state.equals(mock_data.iloc[1, :])


def test_trading_environment_render(mock_data):
    """
    Test the render method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    # The render method is not implemented, so just check that it doesn't raise an error
    env.render()


def test_trading_environment_calculate_fee(mock_data):
    """
    Test the calculate_fee method of the TradingEnvironment class.
    """
    env = TradingEnvironment(mock_data)
    fee = env.calculate_fee(1000000)

    assert fee == 1000000 * 0.0026  # Check that the correct fee tier is used
