import pytest
import numpy as np
from tests.common import BaseSingleAgentTest, EnvironmentConfigs, ObservationValidator

class TestingSingleAgentObservations(BaseSingleAgentTest):
    """Test suite for single agent observation spaces."""

    def get_config(self):
        """Return default Dubins config for observation space testing."""
        return EnvironmentConfigs.get_dubins_config()

    @pytest.mark.parametrize("config_type, expected_size", [
        ("dubins", 5),  # x, y, heading, altitude, airspeed
        ("full", 12),   # x, y, z, roll, pitch, yaw, u, v, w, p, q, r
    ])
    def test_observation_space_dimensions(self, config_type, expected_size):
        """Test observation space dimensions for different aircraft types."""
        if config_type == "dubins":
            config = EnvironmentConfigs.get_dubins_config()
        else:
            config = EnvironmentConfigs.get_full_config()
        env = self.create_env(config)
        assert env.observation_space.shape == (expected_size,)
        assert env.observation_space.dtype == np.float32

    def test_observation_space_sampling(self):
        """Test observation space sampling."""
        observation = self.env.observation_space.sample()
        print(f"Observation: {observation}")
        assert ObservationValidator.validate(
            observation,
            self.env.observation_space.low,
            self.env.observation_space.high
        )

    def test_observation_bounds(self):
        """Test if observations respect their defined bounds."""
        env = self.env
        for _ in range(100):  # Test multiple episodes
            obs, _ = env.reset()
            assert np.all(obs >= env.observation_space.low)
            assert np.all(obs <= env.observation_space.high)

            # Take random action and check next observation
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if not (terminated or truncated):
                assert np.all(obs >= env.observation_space.low)
                assert np.all(obs <= env.observation_space.high)

    @pytest.mark.parametrize("config_type, bound_checks", [
        ("dubins", {
            "heading": (-np.pi, np.pi),
            "altitude": (0, 20000),
            "airspeed": (0, 900)
        }),
        ("full", {
            "roll": (-np.pi, np.pi),
            "pitch": (-0.5 * np.pi, 0.5 * np.pi),
            "yaw": (-np.pi, np.pi),
            "p": (-2*np.pi, 2*np.pi),
            "q": (-2*np.pi, 2*np.pi),
            "r": (-2*np.pi, 2*np.pi)
        })
    ])
    def test_specific_observation_bounds(self, config_type, bound_checks):
        """Test specific bounds for different observation components."""
        if config_type == "dubins":
            config = EnvironmentConfigs.get_dubins_config()
        else:
            config = EnvironmentConfigs.get_full_config()
        env = self.create_env(config)

        # Access the observation object to get the feature mapping
        observation = env.vehicle.observation  # Assuming env.vehicle.observation is an instance of DubinsObservation or FullObservation
        assert hasattr(observation, "features"), "Observation type must define a 'features' attribute"

        # Map feature names to indices
        feature_indices = {feature: idx for idx, feature in enumerate(observation.features)}

        # Validate bounds using the observation space
        obs_space = observation.space
        for feature, (min_val, max_val) in bound_checks.items():
            assert feature in feature_indices, f"Feature '{feature}' not found in observation features"
            idx = feature_indices[feature]

            # Check bounds for the feature
            assert np.isclose(obs_space.low[idx], min_val, atol=1e-5), (f"{feature} lower bound mismatch: expected {min_val}, got {obs_space.low[idx]}")
            assert np.isclose(obs_space.high[idx], max_val, atol=1e-5), (f"{feature} upper bound mismatch: expected {max_val}, got {obs_space.high[idx]}")

    def test_observation_normalization(self):
        """Test if observation normalization works correctly."""
        # Create environment with normalized observations
        config = self.get_config()
        config.update({"normalize_observations": True})
        env = self.create_env(config)

        obs, _ = env.reset()
        print(f"Test Obs: {obs}")
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)


    def test_observation_consistency(self):
        """Test if observations are consistent with environment state."""
        env = self.env
        obs, _ = env.reset()

        # Store initial state
        initial_state = obs.copy()

        # Take no-op action
        zero_action = np.zeros(env.action_space.shape)
        next_obs, _, _, _, _ = env.step(zero_action)

        # Check if state changes are physically reasonable
        dt = env.dt  # assuming environment has a timestep attribute
        max_change = dt * env.max_velocity  # assuming there's a max velocity

        position_changes = np.abs(next_obs[:3] - initial_state[:3])
        assert np.all(position_changes <= max_change), \
            "Position changed too rapidly for given timestep"

    def test_reset_observation_validity(self):
        """Test if reset provides valid initial observations."""
        for _ in range(50):  # Test multiple resets
            obs, info = self.env.reset()

            # Check observation structure
            assert isinstance(obs, np.ndarray)
            assert obs.shape == self.env.observation_space.shape

            # Check observation values
            assert np.all(np.isfinite(obs)), "Reset returned non-finite observation"
            assert np.all(obs >= self.env.observation_space.low)
            assert np.all(obs <= self.env.observation_space.high)

            # Check info dict
            assert isinstance(info, dict)
