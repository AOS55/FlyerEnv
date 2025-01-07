from typing import Optional, Tuple
import gymnasium as gym
import numpy as np

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
        super().__init__(config, render_mode, connection_config)

        if len(self.controlled_vehicles) != 1:
            raise ValueError("SingleAgentEnv must have exactly one controlled vehicle")

        # Setup standard Gym spaces from the Aircraft
        self.vehicle = self.controlled_vehicles[0]
        self.action_space = self.vehicle.action.space
        self.observation_space = self.vehicle.observation.space

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
        processed_action = [float(x) for x in processed_action]

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

        return (
            obs,
            response["reward"],
            response["terminated"],
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

        obs = self.vehicle.observation.observe(response["obs"][self.vehicle.id])
        return obs, response["info"]
