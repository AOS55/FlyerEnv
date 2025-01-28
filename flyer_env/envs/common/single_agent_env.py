from typing import Optional, Tuple
import gymnasium as gym
import numpy as np
import base64

from flyer_env.envs.common.abstract import AbstractEnv, ConnectionConfig

class SingleAgentEnv(AbstractEnv, gym.Env):
    """
    Standard Gymnasium environment for single agent scenarios.
    Implements the gymnasium interface.

    """
    def __init__(
        self,
        config: dict = None,
        render_mode: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None
    ) -> None:

        print(f"render_mode: {render_mode}")
        print(f"config: {config}")
        super().__init__(config, render_mode, connection_config)

        if len(self.controlled_vehicles) != 1:
            raise ValueError("SingleAgentEnv must have exactly one controlled vehicle")
        if self.config.get('max_episode_steps', 0) <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.config.get('time_step', 0) <= 0:
            raise ValueError("time_step must be positive")
        print(f"self.config: {self.config}")

        self.max_velocity = 900.0  # Add reasonable max velocity
        # Setup standard Gym spaces from the Aircraft
        self.vehicle = self.controlled_vehicles[0]
        self.action_space = self.vehicle.action.space
        self.observation_space = self.vehicle.observation.space

        # Private helper methods
        self._observation = self.vehicle.observation
        self._action = self.vehicle.action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Standard Gym step that works with a single Aircraft.
        Converts between Gym interface and Aircraft interface.

        Args:
            action (np.ndarray): Action to take in the environment

        Returns:
            Tuple containing:
                - np.ndarray: Environment observation
                - float: Reward from the action
                - bool: Whether episode is terminated
                - bool: Whether episode is truncated
                - dict: Additional info
        """
        # Process action through Aircraft action space
        processed_action = self.vehicle.action.act(action)

        print(f"single agent action: {processed_action}")

        # Send to server with aircraft ID
        response = self._send_command({
            "Step": {
                "actions": {
                    self.vehicle.id: processed_action
                }
            }
        })

        # Get observation through Aircraft observation space
        obs = self.vehicle.observation.observe(response["obs"][self.vehicle.id])
        reward = response["reward"][self.vehicle.id]
        terminated = response["terminated"][self.vehicle.id]

        print(f"response: {response}")

        return (
            obs,
            reward,
            terminated,
            response["truncated"],
            response["info"]
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset single Aircraft environment.

        Args:
            seed (Optional[int], optional): Random seed for environment. Defaults to None.
            options (Optional[dict], optional): Additional reset options. Defaults to None.

        Returns:
            Tuple containing:
                - np.ndarray: Initial observation
                - dict: Additional info
        """
        response = self._send_command({
            "Reset": {
                "seed": seed
            }
        })
        print(f"Response: {response}")
        obs = self.vehicle.observation.observe(response["obs"][self.vehicle.id])
        return obs, response["info"]

    def render(self):
        """Get RGB array render from the environment."""
        response = self._send_command(command = {"Render": None}, timeout = 10.0)
        print("Command Sent!")

        if "frame" not in response:
            raise RuntimeError("No frame data in render response")

        frame_data = base64.b64decode(response["frame"])
        width = response["width"]
        height = response["height"]

        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((int(height), int(width), 4))
        print(f"frame: {frame}")
        rgb_frame = frame[:, :, :3]

        return rgb_frame
