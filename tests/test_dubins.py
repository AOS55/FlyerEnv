import pytest
import numpy as np
from tests.common import BaseEnvironmentTest, EnvironmentConfigs, ObservationValidator, StateValidator

class TestDubinsEnvironment(BaseEnvironmentTest):
    def get_config(self):
        """Override to use Dubins config"""
        return EnvironmentConfigs.get_dubins_config()

    def test_reset(self):
        """Test basic reset functionality"""
        obs, info = self.env.reset()
        
        # Validate initial observation
        expected_features = ["x", "y", "heading", "altitude", "airspeed"]
        assert ObservationValidator.validate(obs, expected_features)
        
        # Test deterministic reset with seed
        seed = 42
        obs1, _ = self.env.reset(seed=seed)
        obs2, _ = self.env.reset(seed=seed)
        
        assert StateValidator.check_consistency(obs1, obs2)

    def test_step_response(self):
        """Test basic step functionality"""
        obs, _ = self.env.reset()
        
        # Test a zero action first
        action = np.zeros(3)  # [acceleration, bank_angle, vertical_speed]
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Validate response
        expected_features = ["x", "y", "heading", "altitude", "airspeed"]
        assert ObservationValidator.validate(obs, expected_features)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated

    def test_action_limits(self):
        """Test response to extreme actions"""
        self.env.reset()
        
        # Test max positive action
        max_action = np.ones(3)
        obs, reward, terminated, truncated, info = self.env.step(max_action)
        assert not terminated, "Environment terminated on max action"
        
        # Test max negative action
        min_action = -np.ones(3)
        obs, reward, terminated, truncated, info = self.env.step(min_action)
        assert not terminated, "Environment terminated on min action"

    def test_episode_completion(self):
        """Test full episode execution"""
        metrics = self.run_episode(steps=100)
        
        assert metrics['steps'] > 0, "Episode had zero steps"
        assert isinstance(metrics['total_reward'], float)
        
        # If episode terminated, check it was for valid reason
        if metrics['terminated']:
            assert metrics['steps'] < 100, "Episode terminated after max steps"

    @pytest.mark.parametrize("invalid_action", [
        np.array([np.inf, 0, 0]),  # infinite value
        np.array([np.nan, 0, 0]),  # NaN value
        np.array([0, 0]),          # wrong size
    ])
    def test_invalid_actions(self, invalid_action):
        """Test handling of invalid actions"""
        self.env.reset()
        
        with pytest.raises(Exception):
            self.env.step(invalid_action)