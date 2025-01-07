import pytest
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

from flyer_env.envs.common.abstract import AbstractEnv, ConnectionConfig
from .configs import EnvironmentConfigs
from .validators import ObservationValidator, ActionValidator, StateValidator

class BaseEnvironmentTest(ABC):
    """
    Abstract base class for all environment tests.
    Provides common setup, teardown, and utility methods.
    """

    @pytest.fixture(autouse=True)
    def setup_env(self, render_mode: Optional[str] = None):
        """
        Setup fixture that runs automatically before each test.
        Creates and configures the environment.

        Args:
            render_mode: Optional rendering mode for visualization
        """
        self.env = None
        try:
            config = self.get_config()
            if render_mode:
                config = EnvironmentConfigs.modify_config(
                    config,
                    {"agent_config": {"mode": render_mode}}
                )
            self.env = self.create_env(config)
            yield
        finally:
            if self.env:
                self.env.close()

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get environment configuration. Must be implemented by child classes.

        Returns:
            Dict containing environment configuration
        """
        pass

    @abstractmethod
    def create_env(self, config: Optional[Dict[str, Any]] = None) -> AbstractEnv:
        """
        Create environment instance. Must be implemented by child classes.

        Args:
            config: Optional configuration dictionary

        Returns:
            Configured environment instance
        """
        pass

    def validate_observation(self, obs: Dict[str, Any], expected_features: list) -> bool:
        """
        Validate observation structure and content.

        Args:
            obs: Observation to validate
            expected_features: List of expected observation features

        Returns:
            True if validation passes
        """
        return ObservationValidator.validate(obs, expected_features)

    def validate_action(self, action: np.ndarray, action_space) -> bool:
        """
        Validate action against action space constraints.

        Args:
            action: Action to validate
            action_space: Space defining valid actions

        Returns:
            True if validation passes
        """
        return ActionValidator.validate(action, action_space.low, action_space.high)

    def validate_state_consistency(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Check consistency between two states within tolerance.

        Args:
            state1: First state to compare
            state2: Second state to compare
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            True if states are consistent within tolerance
        """
        return StateValidator.check_consistency(state1, state2, rtol=rtol, atol=atol)

    def run_episode(
        self,
        max_steps: int = 100,
        seed: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete episode with optional rendering.

        Args:
            max_steps: Maximum number of steps to run
            seed: Optional random seed
            render: Whether to render the environment

        Returns:
            Dictionary containing episode metrics
        """
        metrics = {
            'total_reward': 0,
            'steps': 0,
            'terminated': False,
            'truncated': False
        }

        obs, info = self.env.reset(seed=seed)

        for _ in range(max_steps):
            if render:
                self.env.render()

            action = self.get_action()
            obs, reward, terminated, truncated, info = self.env.step(action)

            metrics['total_reward'] += reward
            metrics['steps'] += 1

            if terminated or truncated:
                metrics['terminated'] = terminated
                metrics['truncated'] = truncated
                break

        return metrics

    @abstractmethod
    def get_action(self) -> Any:
        """
        Get action for environment step. Must be implemented by child classes.

        Returns:
            Valid action for the environment
        """
        pass

    def test_env_creation(self):
        """Test basic environment creation and initialization."""
        assert self.env is not None
        assert isinstance(self.env, AbstractEnv)

    def test_env_seeding(self):
        """Test environment seeding and determinism."""
        seed = 42
        obs1, _ = self.env.reset(seed=seed)
        obs2, _ = self.env.reset(seed=seed)
        obs3, _ = self.env.reset(seed=seed)
        obs4, _ = self.env.reset(seed=seed)
        print(f"Observation 1: {obs1}, Observation 2: {obs2}, Observation 3: {obs3}, Observation 4: {obs4}")

        assert self.validate_state_consistency(obs1, obs2)

    def test_env_closing(self):
        """Test proper environment cleanup."""
        self.env.close()
        assert not hasattr(self.env, '_connected') or not self.env._connected
