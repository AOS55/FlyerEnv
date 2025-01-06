import pytest
import numpy as np
from tests.common import BaseEnvironmentTest, EnvironmentConfigs, ActionValidator

class TestActionSpaces(BaseEnvironmentTest):
    @pytest.mark.parametrize("config_type, expected_size", [
        ("dubins", 3),  # acceleration, bank_angle, vertical_speed
        ("full", 4),    # elevator, aileron, throttle, rudder
    ])
    def test_action_space_dimensions(self, config_type, expected_size):
        """Test action space dimensions for different aircraft types"""
        if config_type == "dubins":
            config = EnvironmentConfigs.get_dubins_config()
        else:
            config = EnvironmentConfigs.get_full_config()
            
        env = self.create_env(config)
        
        assert env.action_space.shape == (expected_size,)
        assert env.action_space.dtype == np.float32

    def test_action_space_sampling(self):
        """Test action space sampling"""
        action = self.env.action_space.sample()
        
        # Validate sample is within bounds
        assert ActionValidator.validate(
            action,
            self.env.action_space.low,
            self.env.action_space.high
        )

    def test_continuous_action_processing(self):
        """Test processing of continuous actions"""
        self.env.reset()
        
        # Test various action magnitudes
        test_actions = [
            np.zeros(3),              # neutral
            np.ones(3) * 0.5,         # moderate
            np.ones(3),               # maximum
            -np.ones(3),              # minimum
            np.array([1, -1, 0.5]),   # mixed
        ]
        
        for action in test_actions:
            obs, reward, terminated, truncated, info = self.env.step(action)
            assert not terminated, f"Environment terminated on valid action: {action}"

    def test_action_clipping(self):
        """Test if out-of-bounds actions are properly clipped"""
        self.env.reset()
        
        # Try action beyond bounds
        large_action = np.ones(3) * 2.0
        obs, reward, terminated, truncated, info = self.env.step(large_action)
        
        # Environment should handle this without crashing
        assert not terminated, "Environment terminated on large action"

    def test_action_response_correlation(self):
        """Test if actions produce correlated state changes"""
        obs, _ = self.env.reset()
        initial_state = obs.copy()
        
        # Test positive vertical speed
        action = np.array([0, 0, 1.0])  # max positive vertical speed
        obs, _, _, _, _ = self.env.step(action)
        assert obs['altitude'] > initial_state['altitude'], "Altitude did not increase with positive vertical speed"
        
        # Test negative vertical speed
        action = np.array([0, 0, -1.0])  # max negative vertical speed
        obs, _, _, _, _ = self.env.step(action)
        assert obs['altitude'] < initial_state['altitude'], "Altitude did not decrease with negative vertical speed"

class TestMultiAgentActions(BaseEnvironmentTest):
    def get_config(self):
        return EnvironmentConfigs.get_multi_agent_config()

    def test_multi_agent_action_spaces(self):
        """Test action spaces in multi-agent setting"""
        obs, _ = self.env.reset()
        
        # Check each agent has correct action space
        for agent_id, agent_obs in obs.items():
            action = self.env.action_space[agent_id].sample()
            assert ActionValidator.validate(
                action,
                self.env.action_space[agent_id].low,
                self.env.action_space[agent_id].high
            )

    def test_independent_agent_control(self):
        """Test agents can be controlled independently"""
        obs, _ = self.env.reset()
        
        # Create different actions for each agent
        actions = {
            agent_id: self.env.action_space[agent_id].sample()
            for agent_id in obs.keys()
        }
        
        # Step environment with different actions
        new_obs, reward, terminated, truncated, info = self.env.step(actions)
        
        # Verify each agent received its observation
        assert set(new_obs.keys()) == set(obs.keys())