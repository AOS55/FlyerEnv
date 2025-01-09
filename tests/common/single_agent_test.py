import pytest
import numpy as np
from typing import Dict, Any, Optional, List

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
from tests.common.base_test import BaseEnvironmentTest

class BaseSingleAgentTest(BaseEnvironmentTest):
    """
    Base class for single agent environment tests.
    Implements specific validation and testing methods for single agent scenarios.
    """

    def create_env(self, config: Optional[Dict[str, Any]] = None) -> SingleAgentEnv:
        """
        Create single agent environment instance.

        Args:
            config: Optional configuration dictionary

        Returns:
            Configured SingleAgentEnv instance
        """
        return SingleAgentEnv(config=config)

    def get_action(self) -> np.ndarray:
        """
        Get random action from environment's action space.

        Returns:
            Valid action for single agent
        """
        return self.env.action_space.sample()

    def test_single_agent_spaces(self):
        """Test action and observation space specifications."""
        # Verify action space
        assert hasattr(self.env, 'action_space'), "Environment missing action_space"
        assert hasattr(self.env.action_space, 'shape'), "Action space missing shape"
        assert hasattr(self.env.action_space, 'dtype'), "Action space missing dtype"

        # Verify observation space
        assert hasattr(self.env, 'observation_space'), "Environment missing observation_space"
        assert hasattr(self.env.observation_space, 'shape'), "Observation space missing shape"
        assert hasattr(self.env.observation_space, 'dtype'), "Observation space missing dtype"

    def test_single_step_execution(self):
        """Test single step functionality and output format."""
        obs, info = self.env.reset()
        action = self.get_action()

        # Execute step
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Verify types
        assert isinstance(next_obs, (np.ndarray, dict)), "Invalid observation type"
        assert isinstance(reward, (int, float)), "Invalid reward type"
        assert isinstance(terminated, bool), "Invalid terminated flag type"
        assert isinstance(truncated, bool), "Invalid truncated flag type"
        assert isinstance(info, dict), "Invalid info type"

        # Verify observation shape matches space
        if isinstance(next_obs, np.ndarray):
            assert next_obs.shape == self.env.observation_space.shape

        # Verify reward is finite
        assert np.isfinite(reward), "Reward is not finite"

    @pytest.mark.parametrize("invalid_action", [
        None,  # None value
        "invalid",  # Wrong type
        np.array([]),  # Empty array
        np.array([np.inf]),  # Infinite value
        np.array([np.nan]),  # NaN value
    ])
    def test_invalid_actions(self, invalid_action):
        """Test environment's response to invalid actions."""
        self.env.reset()
        with pytest.raises(Exception):
            self.env.step(invalid_action)

    def test_reset_state(self):
        """Test environment reset functionality."""
        initial_obs, initial_info = self.env.reset()

        # Run some steps
        for _ in range(10):
            action = self.get_action()
            self.env.step(action)

        # Reset and compare
        reset_obs, reset_info = self.env.reset()

        # Verify observation structure remains consistent
        assert type(initial_obs) == type(reset_obs), "Reset observation type mismatch"
        if isinstance(initial_obs, np.ndarray):
            assert initial_obs.shape == reset_obs.shape, "Reset observation shape mismatch"

        # Verify info structure remains consistent
        assert set(initial_info.keys()) == set(reset_info.keys()), "Reset info keys mismatch"

    def test_episode_termination(self):
        """Test proper episode termination."""
        metrics = self.run_episode(max_steps=1000)

        # Episode should either complete normally or terminate early
        assert metrics['steps'] > 0, "Episode had zero steps"
        assert metrics['steps'] <= 1000, "Episode exceeded max steps"

        if metrics['terminated']:
            assert metrics['steps'] < 1000, "Episode terminated at max steps"

        if metrics['truncated']:
            assert metrics['steps'] == 1000, "Episode truncated before max steps"

    def test_state_consistency(self):
        """Test physical consistency of state transitions."""
        obs, _ = self.env.reset()
        initial_state = obs.copy()

        action = self.get_action()
        next_obs, _, _, _, _ = self.env.step(action)

        # Verify state changes are physically reasonable
        if isinstance(next_obs, np.ndarray):
            state_diff = np.abs(next_obs - initial_state)
            assert np.all(np.isfinite(state_diff)), "Non-finite state changes"
            assert np.all(state_diff < 1e6), "Unreasonably large state changes"

    def verify_observation_features(self, obs: np.ndarray, expected_features: List[str]):
        """
        Verify observation contains expected features.

        Args:
            obs: Observation to verify
            expected_features: List of expected feature names

        Returns:
            True if validation passes
        """
        return self.validate_observation(obs, expected_features)

    def test_reproducibility(self):
        """Test environment reproducibility with same seed."""
        seed = 12345
        n_steps = 100

        # First run
        self.env.reset(seed=seed)
        actions = []
        observations1 = []

        for _ in range(n_steps):
            action = self.get_action()
            actions.append(action)
            obs, _, _, _, _ = self.env.step(action)
            observations1.append(obs)

        # Second run
        self.env.reset(seed=seed)
        observations2 = []

        for action in actions:
            obs, _, _, _, _ = self.env.step(action)
            observations2.append(obs)

        # Verify observations match
        for i in range(n_steps):
            print(f" obs1: {observations1[i]}\n obs2: {observations2[i]}")
            assert np.allclose(observations1[i], observations2[i], rtol=1e-2, atol=1e-2), f"Non-deterministic behavior at step {i}"
