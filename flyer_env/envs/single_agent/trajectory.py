# envs/single_agent/trajectory.py

from typing import Dict, Any, Optional, List, Literal, Tuple
import numpy as np
import logging

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
# Import NEW config definitions and helpers
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    StartConfig, TaskConfig, TrajectoryTaskConfigData, AgentRenderConfig, Vec3,
    MotionPrimitiveConfig, StraightAndLevelMotionConfig, CoordinatedTurnMotionConfig # Import motion types
    # Import Climb, Descend, StraightAndTurn motion configs when defined
)
from flyer_env.envs.common.config_utils import (
    DubinsAircraftPresetBuilder, # Trajectory usually uses Dubins
    FullAircraftPresetBuilder,
    build_fixed_start_config # Trajectory usually uses fixed start
)

logger = logging.getLogger(__name__)

class TrajectoryFlyerEnv(SingleAgentEnv):
    """
    Environment for executing specific motion primitives.
    Uses structured configuration (SingleAgentEnvConfig).
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        # --- Trajectory Task Parameters ---
        motion_type: str = "straight_and_level",
        target_velocity: float = 30.0,
        motion_params: Optional[Dict[str, Any]] = None,
        # --- Start Position ---
        start_position: Tuple[float, float, float] = (0.0, 0.0, -100.0), # NED
        start_heading_deg: float = 0.0, # Degrees
        # --- Aircraft & Env Params ---
        use_full_aircraft: bool = False,
        # Usually Dubins for trajectory tasks
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        time_step: float = 1/60.0,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
    ) -> None:
        """
        Initialize trajectory environment using structured config.

        Args:
            motion_type: Type of motion primitive (e.g., "straight_and_level", "coordinated_turn").
            target_velocity: Desired airspeed for the trajectory (m/s).
            motion_params: Dictionary of parameters specific to the motion_type.
                            e.g., {"target_distance": 1000.0} for straight_and_level
                            e.g., {"turn_radius": 200.0, "turn_angle": 90.0, "direction": "Right"} for coordinated_turn
            start_position: Fixed starting position (x, y, z NED).
            start_heading_deg: Fixed starting heading (degrees from North).
            seed: Random seed.
            render_mode: "human" or "rgb_array".
            max_episode_steps: Max steps per episode.
            time_step: Simulation time step.
            normalize_observations: Global normalization flag.
            normalize_actions: Global normalization flag.
        """
        if motion_params is None:
            motion_params = {}

        # --- 1. Build Motion Primitive Config ---
        motion_config: MotionPrimitiveConfig
        if motion_type == "straight_and_level":
            motion_config = StraightAndLevelMotionConfig(
                target_distance=float(motion_params.get("target_distance", 1000.0))
            )
        elif motion_type == "coordinated_turn":
             # Convert turn angle from degrees (if provided) to radians
             turn_angle_deg = float(motion_params.get("turn_angle", 90.0))
             motion_config = CoordinatedTurnMotionConfig(
                 turn_radius=float(motion_params.get("turn_radius", 200.0)),
                 turn_angle=np.radians(turn_angle_deg), # Store as radians
                 direction=motion_params.get("direction", "Right")
             )
        # elif motion_type == "climb":
        #     # Define ClimbMotionConfig and instantiate here
        #     pass
        # elif motion_type == "descend":
        #     # Define DescendMotionConfig and instantiate here
        #     pass
        # elif motion_type == "straight_and_turn":
        #      # Define StraightAndTurnMotionConfig and instantiate here
        #      pass
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")


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
        )

        # --- 3. Build Aircraft Physics Config (Dubins) ---
        if use_full_aircraft:
            # Use Full aircraft model if requested
            physics_config = FullAircraftPresetBuilder.get_physics_config() # Or add ac_type if needed
            aircraft_type_str = "full"
            # You might need to adjust start_config for the full model if Dubins params aren't sufficient
            # e.g., Does the 'speed' param in build_fixed_start_config correspond to 'u' or total airspeed?
            # For now, assume build_fixed_start_config works okay.
        else:
            # Default to Dubins
            physics_config = DubinsAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "dubins"

        # --- 4. Build Start Config (Fixed) ---
        start_config = build_fixed_start_config(
            position=start_position,
            speed=target_velocity, # Start at the target trajectory speed
            heading_deg=start_heading_deg
        )

        # --- 5. Build Task Config ---
        task_data = TrajectoryTaskConfigData(
            motion=motion_config,
            target_velocity=float(target_velocity)
        )
        task_config = TaskConfig(type="Trajectory", config=task_data)

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

    # REMOVE private _build_motion_config helper method
    # REMOVE ALL @classmethod builders (build_straight_and_level, etc.)
