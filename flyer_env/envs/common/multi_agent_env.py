from typing import Optional, Tuple, Dict
import gymnasium as gym
import numpy as np

from flyer_env.envs.common.abstract import AbstractEnv, ConnectionConfig


class MultiAgentEnv(AbstractEnv):
    """
    MultiAgent Gymnasium type environment for multiple agent scenarios.
    Based on the PettingZoo interface.

    """
    def __init__(
        self,
        config: dict = None,
        render_mode: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None
    ) -> None:
        super().__init__(config, render_mode, connection_config)

        if len(self.controlled_vehicles) == 0:
            raise ValueError("MultiAgentEnv must have at least one controlled vehicle")

    @property
    def observation_spaces(self) -> Dict[str, gym.Space]:
        """
        Return a dict of observation spaces keyed for the aircraft's unique ID.
        """
        return {
            vehicle.id: vehicle.observation.space()
            for vehicle in self.controlled_vehicles
        }

    @property
    def action_spaces(self) -> Dict[str, gym.Space]:
        """
        Return a dict of action spaces keyed by the aircraft's unique ID.
        """
        return {
            vehicle.id: vehicle.action.space()
            for vehicle in self.controlled_vehicles
        }

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dict, dict, dict, dict]:
        """
        Execute a step in the environment for each controlled aircraft.

        Args:
            actions (Dict[str, np.ndarray]):
                A dict mapping each aircraft ID to its corresponding action.

        Returns:
            obs (Dict[str, np.ndarray]): Observations keyed by aircraft ID
            rewards (Dict[str, float]): Rewards keyed by aircraft ID
            terminations (Dict[str, bool]): Termination flags keyed by aircraft ID
            truncations (Dict[str, bool]): Truncation flags keyed by aircraft ID
            info (Dict[str, dict]): Info keyed by aircraft ID (and/or global info)
        """
        processed_actions = {}
        for aircraft in self.controlled_vehicles:
            if aircraft.id not in actions:
                raise KeyError(f"No action provided for aircraft ID '{aircraft.id}'")
            raw_action = actions[aircraft.id]
            processed_actions[aircraft.id] = aircraft.action.act(raw_action)

        response = self._send_command({
            "Step": {
                "actions": processed_actions
            }
        })

        obs = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        for aircraft in self.controlled_vehicles:
            ac_id = aircraft.id

            raw_obs = response["obs"][ac_id]
            obs[ac_id] = aircraft.observation.observe(raw_obs)

            rewards[ac_id] = response["reward"][ac_id]
            terminations[ac_id] = response["terminated"][ac_id]
            truncations[ac_id] = response["truncated"][ac_id]
            # Info could be global or per-aircraft, adapt as needed:
            infos[ac_id] = response["info"].get(ac_id, {})

        return obs, rewards, terminations, truncations, infos

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None
        ):
            """
            Reset the multi-agent environment.

            Returns:
                obs (Dict[str, np.ndarray]): Observations keyed by aircraft ID
                info (Dict[str, dict]): Additional info keyed by aircraft ID (and/or global info)
            """
            # Send a reset command to the server, optionally passing the seed
            reset_command = {"Reset": {}}
            if seed is not None:
                reset_command["Reset"]["seed"] = seed
            if options is not None:
                reset_command["Reset"]["options"] = options

            response = self._send_command(reset_command)

            # Build the return dictionaries
            obs = {}
            infos = {}

            # response["obs"] is a dict keyed by aircraft ID -> raw observation
            for aircraft in self.controlled_vehicles:
                ac_id = aircraft.id
                raw_obs = response["obs"][ac_id]
                obs[ac_id] = aircraft.observation.observe(raw_obs)

            # Optionally store any global info or per-aircraft info
            if isinstance(response["info"], dict):
                infos = response["info"]

            return obs, infos
