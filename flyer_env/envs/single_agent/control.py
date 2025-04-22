# envs/single_agent/control.py

# --- Imports ---
# Note the change in the config_utils import
from typing import Dict, Any, Optional, Union, Tuple, Literal, get_args # Added get_args
import numpy as np
import logging

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    StartConfig, TaskConfig, ControlTaskConfigData, AgentRenderConfig, Vec3
)
# --- MODIFIED IMPORT ---
from flyer_env.envs.common.config_utils import (
    DubinsAircraftPresetBuilder,
    FullAircraftPresetBuilder, # Import the whole class
    build_random_start_config,
    # build_fixed_start_config # Not used here currently
    # build_trim_condition # REMOVED direct import
)

logger = logging.getLogger(__name__)

class ControlFlyerEnv(SingleAgentEnv):
    """
    Environment for fundamental flight parameter regulation tasks.
    Focuses on precise control of individual state variables.
    Uses structured configuration (SingleAgentEnvConfig).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        control_type: Literal["altitude", "heading", "speed", "pitch", "roll"] = "altitude",
        target_value: float = 1000.0,
        tolerance: float = 10.0,
        start_deviation: Union[float, Tuple[float, float]] = 100.0,
        use_full_aircraft: bool = False,
        trim_condition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        time_step: float = 1/60.0,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
    ) -> None:
        # ... (initial checks remain the same) ...
        if control_type in ["pitch", "roll"] and not use_full_aircraft:
            raise ValueError(f"Control type '{control_type}' requires use_full_aircraft=True")
        if trim_condition_params and not use_full_aircraft:
             logger.warning("Trim condition specified but use_full_aircraft=False. Trim will be ignored.")
             trim_condition_params = None

        # --- 1. Build EnvConfig ---
        agent_render_cfg = AgentRenderConfig()
        if render_mode == "rgb_array":
             agent_render_cfg.mode = "RGBArray"
        elif render_mode == "human":
             agent_render_cfg.mode = "human"

        env_config = EnvConfig(
            seed=seed,
            max_episode_steps=max_episode_steps,
            time_step=time_step,
            agent_config=agent_render_cfg,
            normalize_observations=normalize_observations,
            normalize_actions=normalize_actions,
        )

        # --- 2. Build Aircraft Physics Config ---
        if use_full_aircraft:
            physics_config = FullAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "full"
        else:
            physics_config = DubinsAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "dubins"

        # --- 3. Build Start Config ---
        if isinstance(start_deviation, (int, float)):
            min_dev, max_dev = -abs(start_deviation), abs(start_deviation)
        else:
            min_dev, max_dev = start_deviation

        start_params = {}
        if control_type == "altitude":
            start_params["min_altitude_agl"] = target_value + min_dev
            start_params["max_altitude_agl"] = target_value + max_dev
        elif control_type == "speed":
            start_params["min_speed"] = target_value + min_dev
            start_params["max_speed"] = target_value + max_dev
        elif control_type == "heading":
             target_heading_deg = np.degrees(target_value)
             dev_deg = np.degrees(max_dev)
             start_params["min_heading_deg"] = target_heading_deg - dev_deg
             start_params["max_heading_deg"] = target_heading_deg + dev_deg

        # --- MODIFIED CALL ---
        # Build trim condition dictionary if params are provided using the class method
        trim_cond_dict = None
        if trim_condition_params:
            trim_cond_dict = FullAircraftPresetBuilder.build_trim_condition( # Call via class
                trim_type=trim_condition_params.get("type", "StraightAndLevel"),
                airspeed=trim_condition_params.get("airspeed", 80.0),
                gamma_deg=trim_condition_params.get("gamma_deg"),
                bank_angle_deg=trim_condition_params.get("bank_angle_deg")
            )
        start_params["trim_condition"] = trim_cond_dict

        start_config = build_random_start_config(**start_params)


        # --- 4. Build Task Config ---
        task_control_type_literal = control_type.capitalize()
        valid_control_types = get_args(ControlTaskConfigData.__annotations__['control_type'])
        if task_control_type_literal not in valid_control_types:
             raise ValueError(f"Invalid control_type '{control_type}'. Must be one of {valid_control_types}")

        task_data = ControlTaskConfigData(
            control_type=task_control_type_literal,
            target=float(target_value),
            tolerance=float(tolerance)
        )
        task_config = TaskConfig(type="Control", config=task_data)

        # --- 5. Build Aircraft Definition ---
        aircraft_def = AircraftDefinition(
            type=aircraft_type_str,
            physics=physics_config,
            start=start_config,
            task=task_config,
            normalize_obs=env_config.normalize_observations,
            normalize_act=env_config.normalize_actions
        )

        # --- 6. Build Final Environment Config ---
        final_env_config = SingleAgentEnvConfig(
            env=env_config,
            aircraft=aircraft_def
        )

        # --- 7. Initialize Base Class ---
        super().__init__(config=final_env_config, render_mode=render_mode)
