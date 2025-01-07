import pytest
import numpy as np
from tests.common import BaseSingleAgentTest, EnvironmentConfigs, ActionValidator

class TestSingleAgentActions(BaseSingleAgentTest):
    """Test suite for single agent action spaces."""

    def get_config(self):
        """Return default Dubins config for action space testing."""
        return EnvironmentConfigs.get_dubins_config()

    @pytest.mark.parametrize("config_type, expected_size", [
        ("dubins", 3),  # acceleration, bank_angle, vertical_speed
        ("full", 4),    # elevator, aileron, throttle, rudder
    ])
    def test_action_space_dimensions(self, config_type, expected_size):
        """Test action space dimensions for different aircraft types."""
        if config_type == "dubins":
            config = EnvironmentConfigs.get_dubins_config()
        else:
            config = EnvironmentConfigs.get_full_config()

        env = self.create_env(config)

        assert env.action_space.shape == (expected_size,)
        assert env.action_space.dtype == np.float32

    def test_action_space_sampling(self):
        """Test action space sampling."""
        action = self.env.action_space.sample()

        # Validate sample is within bounds
        assert ActionValidator.validate(
            action,
            self.env.action_space.low,
            self.env.action_space.high
        )

    @pytest.mark.parametrize("action", [
        np.zeros(3),              # neutral
        np.ones(3) * 0.5,         # moderate
        np.ones(3),               # maximum
        -np.ones(3),              # minimum
        np.array([1, -1, 0.5]),   # mixed
    ])
    def test_continuous_action_processing(self, action):
        """Test processing of continuous actions."""
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert not terminated, f"Environment terminated on valid action: {action}"

    def test_action_clipping(self):
        """Test if out-of-bounds actions are properly clipped."""
        self.env.reset()

        # Try action beyond bounds
        large_action = np.ones(self.env.action_space.shape) * 2.0
        obs, reward, terminated, truncated, info = self.env.step(large_action)

        # Environment should handle this without crashing
        assert not terminated, "Environment terminated on large action"

        # Verify observation and reward are valid
        assert isinstance(obs, (np.ndarray, dict)), "Invalid observation type"
        assert isinstance(reward, (int, float)), "Invalid reward type"
        assert np.isfinite(reward), "Non-finite reward received"

    def test_action_response_correlation(self):
        """Test if actions produce correlated state changes."""
        self.env.reset()

        # Define indices for difference observation components
        ALTITUDE_IDX = 3
        AIRSPEED_IDX = 4

        # Add tolerance for small changes
        ALTITUDE_TOLERANCE = 0.1
        SPEED_TOLERANCE = 0.1

        # Test sequence of different actions and their effects
        test_sequences = [
            {
                'action': np.array([0, 0, 1.0]),  # vertical speed up
                'steps': 10,
                'check': lambda old, new: abs(new[ALTITUDE_IDX] - old[ALTITUDE_IDX]) > ALTITUDE_TOLERANCE,
                'message': "Altitude did not increase with positive vertical speed"
            },
            {
                'action': np.array([1.0, 0, 0]),  # acceleration
                'steps': 10,
                'check': lambda old, new: abs(new[AIRSPEED_IDX] - old[AIRSPEED_IDX]) > SPEED_TOLERANCE,
                'message': "Speed did not increase with positive acceleration"
            }
        ]

        for test in test_sequences:
            obs_before = self.env.reset()[0]
            for _ in range(test['steps']):
                obs_after, _, _, _, _ = self.env.step(test['action'])
            assert test['check'](obs_before, obs_after), test['message']
