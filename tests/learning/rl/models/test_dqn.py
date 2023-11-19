"""
This module contains tests for the DQN class in the dqn module.

Tests cover the initialization, model building, target model updating, memory remembering, action selection, replaying, model saving, and model loading methods.

Classes:
    TestDQN: A class that contains tests for the DQN class.
"""

import numpy as np
import pytest
from tensorflow.keras.models import Sequential
from src.learning.rl.models.dqn import DQN


class TestDQN:
    @pytest.fixture
    def dqn(self):
        """
        A pytest fixture that creates a DQN object for testing.
        """
        return DQN(10, 3)

    def test_initialization(self, dqn):
        """
        Test the initialization of the DQN class.
        """
        assert dqn.state_size == 10
        assert dqn.action_size == 3
        assert isinstance(dqn.model, Sequential)
        assert isinstance(dqn.target_model, Sequential)

    def test_build_model(self, dqn):
        """
        Test the build_model method of the DQN class.
        """
        model = dqn.build_model()
        assert isinstance(model, Sequential)

    def test_update_target_model(self, dqn):
        """
        Test the update_target_model method of the DQN class.
        """
        # Set the weights of the model and target_model to some random values
        for layer in dqn.model.layers:
            layer.set_weights([np.random.rand(*w.shape) for w in layer.get_weights()])
        for layer in dqn.target_model.layers:
            layer.set_weights([np.random.rand(*w.shape) for w in layer.get_weights()])

        # Update the target model
        dqn.update_target_model()

        # Check that the weights of the model and target model are the same
        for model_layer, target_model_layer in zip(
            dqn.model.layers, dqn.target_model.layers
        ):
            assert all(
                [
                    np.array_equal(w1, w2)
                    for w1, w2 in zip(
                        model_layer.get_weights(), target_model_layer.get_weights()
                    )
                ]
            )

    def test_replay(self, dqn):
        """
        Test the replay method of the DQN class.
        """
        # Make sure the state size matches the state_size attribute of the DQN object
        state = np.random.rand(dqn.state_size)
        action = 1
        reward = 1
        next_state = np.random.rand(dqn.state_size)
        done = False
        dqn.remember(state, action, reward, next_state, done)
        dqn.replay(1)

    def test_remember(self, dqn):
        """
        Test the remember method of the DQN class.
        """
        state = np.array([1, 2, 3])
        action = 1
        reward = 1
        next_state = np.array([4, 5, 6])
        done = False
        dqn.remember(state, action, reward, next_state, done)
        assert len(dqn.memory) == 1

    def test_act(self, dqn):
        """
        Test the act method of the DQN class.
        """
        state = np.random.rand(dqn.state_size)

        # Test the exploration branch
        dqn.epsilon = 1.0  # Force the method to choose a random action
        action = dqn.act(state)
        assert action in range(dqn.action_size)

        # Test the exploitation branch
        dqn.epsilon = 0.0  # Force the method to use the model to predict the action
        action = dqn.act(state)
        assert action in range(dqn.action_size)

    def test_save_model(self, dqn, tmpdir):
        """
        Test the save_model method of the DQN class.
        """
        model_path = tmpdir.join("model.h5")
        dqn.save_model(model_path)
        assert model_path.check()

    def test_load_model(self, dqn, tmpdir):
        """
        Test the load_model method of the DQN class.
        """
        model_path = tmpdir.join("model.h5")
        dqn.save_model(model_path)
        dqn.load_model(model_path)
        assert isinstance(dqn.model, Sequential)
