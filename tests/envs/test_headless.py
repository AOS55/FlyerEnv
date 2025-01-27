from flyer_env.envs.common.single_agent_env import SingleAgentEnv
import pytest
import numpy as np
from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestHeadlessRenderer(BaseSingleAgentTest):
    """Test suite for headless rendering mode"""

    def get_config(self):
        """Get config with headless rendering enabled"""
        config = EnvironmentConfigs.get_dubins_config()
        config["agent_config"]["mode"] = "RGBArray"
        return config

    def test_headless_initialization(self):
        """Test that headless renderer initializes correctly"""
        env = self.create_env(self.get_config())

        # Verify render mode is set correctly
        assert env.config["agent_config"]["mode"] == "RGBArray"

        # Check render dimensions
        assert env.config["agent_config"]["render_width"] == 800.0
        assert env.config["agent_config"]["render_height"] == 600.0

    def test_frame_generation(self):
        """Test that renderer can generate valid RGB frames."""
        env = self.create_env(self.get_config())
        obs, _ = env.reset()

        # Take a few steps to ensure aircraft is moving
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

            # Get rendered frame
            frame = env.render()

            # Verify frame properties
            assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
            assert frame.dtype == np.uint8, "Frame should be uint8"
            assert len(frame.shape) == 3, "Frame should have 3 dimensions (height, width, channels)"
            assert frame.shape[2] == 3, "Frame should have 3 color channels"

            # Check frame dimensions match config
            assert frame.shape[1] == int(env.config["agent_config"]["render_width"])
            assert frame.shape[0] == int(env.config["agent_config"]["render_height"])

            # Verify frame contains valid pixel values
            assert np.all(frame >= 0) and np.all(frame <= 255), "Pixel values should be in [0, 255]"
            assert not np.all(frame == 0), "Frame should not be completely black"
            assert not np.all(frame == 255), "Frame should not be completely white"


# if __name__=="__main__":

#     from flyer_env.envs.common import SingleAgentEnv

#     config = {
#         "max_episode_steps": 1000,
#         "normalize_actions": True,
#         "steps_per_action": 4,
#         "time_step": 1.0/120.0,
#         "aircraft_config": [{
#             "type": "dubins",
#             "action_type": "Continuous",
#             "observation_type": "Continuous"
#         }],
#         "agent_config": {
#             "render_width": 800.0,
#             "render_height": 600.0,
#             "mode": "human"
#         }
#     }
#     config["agent_config"]["mode"] = "RGBArray"
#     env = SingleAgentEnv(config=config)

#     obs, _ = env.reset()

#     # Take a few steps to ensure aircraft is moving
#     for _ in range(10):
#         action = env.action_space.sample()
#         _, _, terminated, truncated, _ = env.step(action)
#         if terminated or truncated:
#             break

#         # Get rendered frame
#         frame = env.render()

#         # Verify frame properties
#         assert isinstance(frame, np.ndarray), "Frame should be a numpy array"
#         assert frame.dtype == np.uint8, "Frame should be uint8"
#         assert len(frame.shape) == 3, "Frame should have 3 dimensions (height, width, channels)"
#         assert frame.shape[2] == 3, "Frame should have 3 color channels"

#         # Check frame dimensions match config
#         assert frame.shape[1] == int(env.config["agent_config"]["render_width"])
#         assert frame.shape[0] == int(env.config["agent_config"]["render_height"])

#         # Verify frame contains valid pixel values
#         assert np.all(frame >= 0) and np.all(frame <= 255), "Pixel values should be in [0, 255]"
#         assert not np.all(frame == 0), "Frame should not be completely black"
        assert not np.all(frame == 255), "Frame should not be completely white"
