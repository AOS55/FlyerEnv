import pytest
from flyer_env.envs.common.abstract import AbstractEnv
from .configs import EnvironmentConfigs
from .validators import ObservationValidator, ActionValidator, StateValidator
from .test_utils import EpisodeRunner

class BaseEnvironmentTest:
    @pytest.fixture(autouse=True)
    def setup_env(self, render_mode):
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

    def get_config(self):
        """Override this to provide specific config"""
        return EnvironmentConfigs.get_dubins_config()

    def create_env(self, config=None):
        """Create environment with given or default config"""
        return AbstractEnv(config=config)

    def run_episode(self, steps=100):
        """Run a full episode with random actions"""
        return EpisodeRunner.run(self.env, max_steps=steps)