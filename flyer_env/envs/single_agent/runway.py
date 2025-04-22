# envs/single_agent/runway.py

from typing import Dict, Any, Optional, Tuple, Literal # Added Literal
import numpy as np
import logging

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
# Import NEW config definitions and helpers
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    StartConfig, TaskConfig, RunwayTaskConfigData, AgentRenderConfig, Vec3,
    RunwayGenerationConfig # Import runway specific config
)
from flyer_env.envs.common.config_utils import (
    DubinsAircraftPresetBuilder, FullAircraftPresetBuilder,
    build_random_start_config, # Runway env often uses random start near origin
    build_trim_condition
)

logger = logging.getLogger(__name__)


class RunwayFlyerEnv(SingleAgentEnv):
    """
    Environment for runway approach tasks with randomized runway placement.
    Uses structured configuration (SingleAgentEnvConfig).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
         # --- Aircraft Start Randomization ---
        start_altitude_range_m_agl: Tuple[float, float] = (400.0, 600.0), # AGL
        start_airspeed_range_mps: Tuple[float, float] = (70.0, 90.0),    # m/s
        start_heading_range_deg: Tuple[float, float] = (0.0, 360.0),   # Degrees
        start_position_variance: float = 1.0, # Typically start near origin (0,0)
        # --- Runway Randomization Parameters ---
        runway_distance_range_m: Tuple[float, float] = (4000.0, 8000.0),
        runway_bearing_range_deg: Tuple[float, float] = (0.0, 360.0),
        runway_heading_range_deg: Tuple[float, float] = (0.0, 360.0),
        # --- Fixed Runway Geometry ---
        runway_width: float = 45.0,
        runway_length: float = 1800.0,
        glideslope_deg: float = 3.0,
        # --- Aircraft & Env Params ---
        use_full_aircraft: bool = True,
        trim_condition_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 3000,
        time_step: float = 1/60.0,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
        # aircraft_config_override: Optional[Dict[str, Any]] = None, # Optional override physics
    ) -> None:
        """
        Initialize Runway environment using structured config.
        """
        if trim_condition_params and not use_full_aircraft:
             logger.warning("Trim condition specified but use_full_aircraft=False. Trim will be ignored.")
             trim_condition_params = None

        # --- 1. Build Runway Generation Config ---
        # Convert degrees to radians for backend config
        bearing_min_rad = np.radians(runway_bearing_range_deg[0])
        bearing_max_rad = np.radians(runway_bearing_range_deg[1])
        heading_min_rad = np.radians(runway_heading_range_deg[0])
        heading_max_rad = np.radians(runway_heading_range_deg[1])

        runway_gen_config = RunwayGenerationConfig(
            generation_type="random_relative_to_aircraft_start", # Assuming this type
            distance_range=tuple(float(d) for d in runway_distance_range_m),
            bearing_range=(float(bearing_min_rad), float(bearing_max_rad)),
            heading_range=(float(heading_min_rad), float(heading_max_rad)),
            fixed_width=float(runway_width),
            fixed_length=float(runway_length)
            # fixed_position/fixed_heading would be set if generation_type was "fixed"
        )

        # --- 2. Build EnvConfig ---
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
            runway_config=runway_gen_config # Include runway generation config
        )

        # --- 3. Build Aircraft Physics Config ---
        if use_full_aircraft:
            physics_config = FullAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "full"
        else:
            physics_config = DubinsAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "dubins"
        # TODO: Allow overriding specific physics params

        # --- 4. Build Start Config (Random near origin) ---
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
            position_variance=start_position_variance, # Usually low for runway env start
            trim_condition=trim_cond_dict
        )

        # --- 5. Build Task Config ---
        task_data = RunwayTaskConfigData(
            source="dynamic_runway", # Task uses the generated runway
            fixed_width=float(runway_width), # Pass geometry info to task if needed
            fixed_length=float(runway_length),
            fixed_glideslope=np.radians(glideslope_deg) # Pass glideslope in radians
        )
        task_config = TaskConfig(type="Runway", config=task_data)

        # --- 6. Build Aircraft Definition ---
        aircraft_def = AircraftDefinition(
            type=aircraft_type_str,
            physics=physics_config,
            start=start_config,
            task=task_config,
            normalize_obs=env_config.normalize_observations,
            normalize_act=env_config.normalize_actions
        )

        # --- 7. Build Final Environment Config ---
        final_env_config = SingleAgentEnvConfig(
            env=env_config,
            aircraft=aircraft_def
        )

        # --- 8. Initialize Base Class ---
        super().__init__(config=final_env_config, render_mode=render_mode)

    # REMOVE @classmethod builder (build_random_approach)
