import pytest
import numpy as np
from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestEnvironmentConfiguration(BaseSingleAgentTest):
    """Test suite for environment configuration validation and handling."""

    def get_config(self):
        """Return default config for testing."""
        return EnvironmentConfigs.get_dubins_config()

    def test_basic_config_parameters(self):
        """Test that basic configuration parameters are correctly set."""
        config = self.get_config()
        env = self.create_env(config)

        # Test base environment parameters
        assert env.config["max_episode_steps"] == 1000
        assert env.config["steps_per_action"] == 4
        assert np.isclose(env.config["time_step"], 1.0/120.0)

        # Test agent config parameters
        assert env.config["agent_config"]["render_width"] == 800.0
        assert env.config["agent_config"]["render_height"] == 600.0
        assert env.config["agent_config"]["mode"] == "human"

    def test_dubins_aircraft_config(self):
        """Test Dubins aircraft configuration parameters."""
        config = EnvironmentConfigs.get_dubins_config()
        env = self.create_env(config)

        # Get the aircraft config from the environment
        aircraft_config = env.vehicle.config

        # Test aircraft type-specific parameters
        assert aircraft_config["type"] == "dubins"
        assert aircraft_config["action_type"] == "Continuous"
        assert aircraft_config["observation_type"] == "Continuous"

        # Verify state space dimensions
        assert env.observation_space.shape == (5,)  # x, y, heading, altitude, airspeed
        assert env.action_space.shape == (3,)      # acceleration, bank_angle, vertical_speed

        # Test action space bounds
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_full_aircraft_config(self):
        """Test Full aircraft configuration parameters."""
        config = EnvironmentConfigs.get_full_config()
        env = self.create_env(config)

        # Get the aircraft config
        aircraft_config = env.vehicle.config

        # Test aircraft type-specific parameters
        assert aircraft_config["type"] == "full"
        assert aircraft_config["action_type"] == "Continuous"
        assert aircraft_config["observation_type"] == "Continuous"

        # Verify state space dimensions
        assert env.observation_space.shape == (12,)  # x, y, z, roll, pitch, yaw, u, v, w, p, q, r
        assert env.action_space.shape == (4,)       # elevator, aileron, throttle, rudder

    def test_config_modification(self):
        """Test that configuration modifications are properly applied."""
        base_config = self.get_config()

        # Modify configuration
        modified_config = EnvironmentConfigs.modify_config(
            base_config,
            {
                "max_episode_steps": 500,
                "time_step": 1.0/60.0,
                "agent_config": {
                    "render_width": 1024.0,
                    "render_height": 768.0
                }
            }
        )

        env = self.create_env(modified_config)

        # Verify modifications
        assert env.config["max_episode_steps"] == 500
        assert np.isclose(env.config["time_step"], 1.0/60.0)
        assert env.config["agent_config"]["render_width"] == 1024.0
        assert env.config["agent_config"]["render_height"] == 768.0

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration parameters."""
        base_config = self.get_config()

        # Test invalid max_episode_steps
        with pytest.raises(ValueError):
            invalid_config = EnvironmentConfigs.modify_config(
                base_config,
                {"max_episode_steps": -1}
            )
            self.create_env(invalid_config)

        # Test invalid time_step
        with pytest.raises(ValueError):
            invalid_config = EnvironmentConfigs.modify_config(
                base_config,
                {"time_step": 0.0}
            )
            self.create_env(invalid_config)

    def test_config_seeding(self):
        """Test that random seed configuration works correctly."""
        # Create two environments with same seed
        seed = 42
        config1 = EnvironmentConfigs.get_dubins_config(seed=seed)
        config2 = EnvironmentConfigs.get_dubins_config(seed=seed)

        env1 = self.create_env(config1)
        env2 = self.create_env(config2)

        # Reset both environments and compare initial states
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert np.allclose(obs1, obs2)

    def test_observation_normalization(self):
        """Test observation normalization configuration."""
        # Create config with normalized observations
        config = EnvironmentConfigs.modify_config(
            self.get_config(),
            {"normalize_observations": True}
        )

        env = self.create_env(config)
        obs, _ = env.reset()

        # Verify observations are normalized
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_action_normalization(self):
        """Test action normalization configuration."""
        # Create config with normalized actions
        config = EnvironmentConfigs.modify_config(
            self.get_config(),
            {"normalize_actions": True}
        )

        env = self.create_env(config)

        # Verify action space is normalized
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

    def test_config_defaults(self):
        """Test that default configurations are properly set."""
        # Create environment with minimal config
        minimal_config = {
            "aircraft_config": [{
                "type": "dubins",
                "action_type": "Continuous",
                "observation_type": "Continuous"
            }]
        }

        env = self.create_env(minimal_config)

        # Verify defaults
        assert env.config["max_episode_steps"] == 1000
        assert env.config["steps_per_action"] == 4
        assert np.isclose(env.config["time_step"], 1.0/120.0)
        assert env.config["agent_config"]["render_width"] == 800.0
        assert env.config["agent_config"]["render_height"] == 600.0
