# envs/single_agent/landing.py

from typing import Dict, Any, Optional, Tuple, Union, Literal # Added Literal
import numpy as np
import logging

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
# Import NEW config definitions and helpers
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    StartConfig, TaskConfig, LandingTaskConfigData, AgentRenderConfig, Vec3
)
from flyer_env.envs.common.config_utils import (
    DubinsAircraftPresetBuilder, FullAircraftPresetBuilder,
    build_random_start_config, # Landing usually uses random start
    build_trim_condition
)

logger = logging.getLogger(__name__)

# DELETE the _LocalDubinsAircraftPreset and _LocalFullAircraftPreset classes


class LandingFlyerEnv(SingleAgentEnv):
    """
    Environment for forced landing tasks.
    Uses structured configuration (SingleAgentEnvConfig).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        # --- Landing Task Parameters ---
        target_position_coords: Optional[Tuple[float, float, float]] = None, # (x, y, z NED)
        max_landing_speed: float = 25.0,
        max_descent_rate: float = 3.0,
        max_bank_angle_deg: float = 30.0,
        max_landing_distance: float = 200.0, # Lateral tolerance
        landing_complete_height_agl: float = 0.5, # AGL height
        # --- Aircraft Start Randomization ---
        start_altitude_range_m_agl: Tuple[float, float] = (400.0, 600.0), # AGL
        start_airspeed_range_mps: Tuple[float, float] = (70.0, 90.0), # m/s
        start_heading_range_deg: Tuple[float, float] = (0.0, 360.0), # Degrees
        start_position_variance: float = 50.0,
        # --- Aircraft & Env Params ---
        use_full_aircraft: bool = True, # Usually True for landing
        trim_condition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 3000,
        time_step: float = 1/60.0,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
        # aircraft_config_override: Optional[Dict[str, Any]] = None, # Handle via physics_config override?
    ) -> None:
        """
        Initialize Landing environment using structured config.
        """
        if not use_full_aircraft:
             logger.warning("Landing task typically uses the full aircraft model.")
        if trim_condition_params and not use_full_aircraft:
             logger.warning("Trim condition specified but use_full_aircraft=False. Trim will be ignored.")
             trim_condition_params = None

        # --- 1. Build EnvConfig ---
        agent_render_cfg = AgentRenderConfig()
        if render_mode == "rgb_array": agent_render_cfg.mode = "RGBArray"
        elif render_mode == "human": agent_render_cfg.mode = "human"

        env_config = EnvConfig(
            seed=seed,
            max_episode_steps=max_episode_steps,
            time_step=time_step,
            agent_config=agent_render_cfg,
            normalize_observations=normalize_observations,
            normalize_actions=normalize_actions,
        )

        # --- 2. Build Aircraft Physics Config ---
        # Allow overriding ac_type if needed
        ac_type_override = None
        # if aircraft_config_override and 'config' in aircraft_config_override:
        #      ac_type_override = aircraft_config_override['config'].get('ac_type')

        if use_full_aircraft:
            physics_config = FullAircraftPresetBuilder.get_physics_config(
                 ac_type=ac_type_override or "twin_otter" # Default if not overridden
            )
            aircraft_type_str = "full"
        else:
            physics_config = DubinsAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "dubins"
        # TODO: Allow overriding specific physics params if needed

        # --- 3. Build Start Config (Random) ---
        trim_cond_dict = None
        if trim_condition_params:
            trim_cond_dict = FullAircraftPresetBuilder.build_trim_condition(
                trim_type=trim_condition_params.get("type", "StraightAndLevel"),
                airspeed=trim_condition_params.get("airspeed", 80.0),
                gamma_deg=trim_condition_params.get("gamma_deg"),
                bank_angle_deg=trim_condition_params.get("bank_angle_deg")
            )

        start_config = build_random_start_config(
            min_altitude_agl=start_altitude_range_m_agl[0],
            max_altitude_agl=start_altitude_range_m_agl[1],
            min_speed=start_airspeed_range_mps[0],
            max_speed=start_airspeed_range_mps[1],
            min_heading_deg=start_heading_range_deg[0],
            max_heading_deg=start_heading_range_deg[1],
            position_variance=start_position_variance,
            trim_condition=trim_cond_dict
        )

        # --- 4. Build Task Config ---
        target_pos_dataclass = None
        if target_position_coords:
            target_pos_dataclass = Vec3(
                 x=target_position_coords[0],
                 y=target_position_coords[1],
                 z=target_position_coords[2]
            )

        task_data = LandingTaskConfigData(
            target_position=target_pos_dataclass,
            max_landing_speed=float(max_landing_speed),
            max_descent_rate=float(max_descent_rate),
            max_bank_angle=np.radians(max_bank_angle_deg), # Convert to radians
            max_landing_distance=float(max_landing_distance),
            landing_complete_height=float(landing_complete_height_agl) # Assume AGL for this param
        )
        task_config = TaskConfig(type="Landing", config=task_data)

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

    # REMOVE ALL @classmethod builders (build_forced_landing, build_precision_landing)
