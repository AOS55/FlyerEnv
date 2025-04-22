from typing import Optional, Tuple, Any, Dict
import gymnasium as gym
import numpy as np
import base64
from dataclasses import asdict, is_dataclass

from flyer_env.envs.common.abstract import AbstractEnv, ConnectionConfig
from flyer_env.envs.common.config_defs import SingleAgentEnvConfig, AircraftDefinition, EnvConfig


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively converts a dataclass instance to a dictionary."""
    if is_dataclass(obj):
        result = {}
        for f in obj.__dataclass_fields__:
            value = getattr(obj, f)
            result[f] = _dataclass_to_dict(value) # Recurse
        return result
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'): # namedtuple
        return type(obj)(*[_dataclass_to_dict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_dataclass_to_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((_dataclass_to_dict(k), _dataclass_to_dict(v)) for k, v in obj.items())
    else:
        return obj


class SingleAgentEnv(AbstractEnv, gym.Env):
    """
    Standard Gymnasium environment for single agent scenarios.
    Implements the gymnasium interface.

    """
    def __init__(
        self,
        # Expect the structured config object instead of a raw dict
        config: SingleAgentEnvConfig,
        render_mode: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None,
        debug_level: str = "warn"
    ) -> None:

        # print(f"render_mode: {render_mode}")
        # print(f"config: {config}")
        config_dict = self._convert_structured_config_to_dict(config)
        super().__init__(config_dict, render_mode, connection_config, debug_level)

        if len(self.controlled_vehicles) != 1:
            raise ValueError("SingleAgentEnv must have exactly one controlled vehicle")
        # if self.config.get('max_episode_steps', 0) <= 0:
        #     raise ValueError("max_episode_steps must be positive")
        # if self.config.get('time_step', 0) <= 0:
        #     raise ValueError("time_step must be positive")
        # print(f"self.config: {self.config}")

        self.max_velocity = 900.0  # Add reasonable max velocity
        # Setup standard Gym spaces from the Aircraft
        self.vehicle = self.controlled_vehicles[0]
        self.action_space = self.vehicle.action.space
        self.observation_space = self.vehicle.observation.space

        # Private helper methods
        self._observation = self.vehicle.observation
        self._action = self.vehicle.action

    def _convert_structured_config_to_dict(self, structured_config: SingleAgentEnvConfig) -> Dict[str, Any]:
        """
        Converts the SingleAgentEnvConfig dataclass back into the nested
        dictionary format expected by the current AbstractEnv and Rust backend.
        """
        # Start with environment base config
        legacy_dict = _dataclass_to_dict(structured_config.env)

        # Prepare the aircraft_config list (with only one aircraft)
        aircraft_def = structured_config.aircraft
        legacy_aircraft_dict = {
            "id": aircraft_def.id, # Pass ID if needed by Rust config loader
            "type": aircraft_def.type,
            "config": _dataclass_to_dict(aircraft_def.physics),
            "start_config": _dataclass_to_dict(aircraft_def.start),
            "task_config": _dataclass_to_dict(aircraft_def.task),
            "action_type": aircraft_def.action_type,
            "observation_type": aircraft_def.observation_type,
            "normalize_obs": aircraft_def.normalize_obs, # Ensure these flags are passed if Rust uses them
            "normalize_act": aircraft_def.normalize_act,
        }
        # Ensure nested 'config' keys match what Rust expects, e.g., task["config"]
        if 'config' not in legacy_aircraft_dict['task_config']:
                legacy_aircraft_dict['task_config'] = {'config': legacy_aircraft_dict['task_config']}
        if 'config' not in legacy_aircraft_dict['start_config']:
                legacy_aircraft_dict['start_config'] = {'config': legacy_aircraft_dict['start_config']}


        legacy_dict["aircraft_config"] = [legacy_aircraft_dict]

        # Handle optional runway config for RunwayEnv specifically
        if structured_config.env.runway_config:
                legacy_dict["runway_config"] = _dataclass_to_dict(structured_config.env.runway_config)

        # Merge agent_config (render settings) into the top level if AbstractEnv expects it there
        # Or adjust AbstractEnv to look inside legacy_dict['agent_config']
        agent_render_config = legacy_dict.pop('agent_config', {})
        legacy_dict.update(agent_config=agent_render_config) # Add it back at top level

        # Ensure normalize flags from EnvConfig are present at top level if AbstractEnv uses them
        # These might override per-aircraft flags depending on AbstractEnv logic
        legacy_dict["normalize_observations"] = structured_config.env.normalize_observations
        legacy_dict["normalize_actions"] = structured_config.env.normalize_actions


        # Important: Verify the final dictionary structure matches exactly
        # what the Rust backend's `Initialize::config` deserialization expects.
        # Adjust the conversion logic above as needed.

        return legacy_dict

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Standard Gym step that works with a single Aircraft.
        Converts between Gym interface and Aircraft interface.
        """
        # Process action through Aircraft action space
        processed_action = self._action.act(action) # Use stored reference

        # Send to server with aircraft ID
        response = self._send_command({
            "Step": {
                "actions": {
                    self.vehicle.id: processed_action
                }
            }
        })

        # Get observation through Aircraft observation space
        obs = self._observation.observe(response["obs"][self.vehicle.id]) # Use stored reference
        reward = response["reward"][self.vehicle.id]
        terminated = response["terminated"][self.vehicle.id]
        truncated = response["truncated"] # Truncation is global
        info = response["info"]

        return obs, reward, terminated, truncated, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None
        ) -> Tuple[np.ndarray, dict]:
            """
            Reset single Aircraft environment.
            """

            # Pass seed and options to the Reset command
            reset_payload = {"seed": seed}
            if options:
                reset_payload["options"] = options # Passthrough options dict

            response = self._send_command({"Reset": reset_payload})

            # AbstractEnv._initialize_env is NOT called again on Reset
            # We just get the new initial state from the response
            obs_dict = response["obs"][self.vehicle.id]
            obs = self._observation.observe(obs_dict) # Use stored reference
            info = response["info"]

            return obs, info

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
