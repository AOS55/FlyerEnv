from flyer_env.envs.common.observation import DubinsObservation, FullObservation
from flyer_env.envs.common.action import DubinsContinuousAction, FullContinuousAction
import pytest
import numpy as np
import time
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
        aircraft_config = env.vehicle

        # Test aircraft type-specific parameters
        assert aircraft_config.type == "Dubins"
        assert isinstance(aircraft_config.action, DubinsContinuousAction)
        assert isinstance(aircraft_config.observation, DubinsObservation)

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
        aircraft_config = env.vehicle

        # Test aircraft type-specific parameters
        assert aircraft_config.type == "Full"
        assert isinstance(aircraft_config.action, FullContinuousAction)
        assert isinstance(aircraft_config.observation, FullObservation)

        # Verify state space dimensions
        assert np.all(env.action_space.low == -1.0)  # x, y, z, roll, pitch, yaw, u, v, w, p, q, r
        assert np.all(env.action_space.high == 1.0)  # elevator, aileron, throttle, rudder

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

        # TODO: Pass seed in here and see if it follows the environment reset
        # Reset both environments and compare initial states
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)

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

    def test_multiple_environments(self):
        """Test that multiple environments can run simultaneously."""
        # Create two environments with different seeds
        config1 = EnvironmentConfigs.get_dubins_config(seed=42)
        config2 = EnvironmentConfigs.get_dubins_config(seed=43)

        env1 = self.create_env(config1)
        env2 = self.create_env(config2)

        try:
            # Reset both environments
            obs1, _ = env1.reset()
            obs2, _ = env2.reset()

            # Verify they have different initial states due to different seeds
            assert not np.allclose(obs1, obs2), "Environments should have different initial states with different seeds"

            # Run steps in both environments
            for _ in range(100):
                # Take actions in both environments
                action = np.array([0.5, 0, 0])  # Simple test action

                next_obs1, reward1, term1, trunc1, _ = env1.step(action)
                next_obs2, reward2, term2, trunc2, _ = env2.step(action)

                # Verify both environments are still running
                assert not (term1 or trunc1), "Environment 1 terminated unexpectedly"
                assert not (term2 or trunc2), "Environment 2 terminated unexpectedly"

                # Verify observations are different between environments
                assert not np.allclose(next_obs1, next_obs2), "Environments should maintain different states"

        finally:
            # Clean up
            env1.close()
            env2.close()

    def test_long_evaluation_episode(self):
        """Test environment can handle long evaluation episodes without timeout."""
        # Create environments
        config = EnvironmentConfigs.get_dubins_config(seed=42)
        env = self.create_env(config)
        eval_env = self.create_env(config)

        try:
            # Run full length episode in both envs
            for env_name, test_env in [("train", env), ("eval", eval_env)]:
                obs, _ = test_env.reset()

                # Run for length of typical episode
                steps = 0
                max_steps = 3000

                for _ in range(max_steps):
                    action = np.array([0.5, 0, 0])  # Simple test action
                    start_time = time.time()

                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    step_time = time.time() - start_time

                    print(f"{env_name} Step {steps}: {step_time:.3f}s")
                    steps += 1

                    if terminated or truncated:
                        break

                print(f"{env_name} completed {steps} steps")
                assert steps > 0, f"{env_name} environment failed to complete any steps"

        finally:
            env.close()
            eval_env.close()
