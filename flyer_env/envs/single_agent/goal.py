# envs/single_agent/goal.py

from typing import Dict, Any, Optional, List, Tuple, Literal # Added Literal
import numpy as np
import logging # Import logging

from flyer_env.envs.common.single_agent_env import SingleAgentEnv
# Import NEW config definitions and helpers
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    StartConfig, TaskConfig, GoalTaskConfigData, AgentRenderConfig, Vec3
)
from flyer_env.envs.common.config_utils import (
    DubinsAircraftPresetBuilder, FullAircraftPresetBuilder,
    build_fixed_start_config # Goal env often uses a fixed start
)

logger = logging.getLogger(__name__)

class GoalFlyerEnv(SingleAgentEnv):
    """Environment for 3D waypoint navigation tasks."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        # --- Goal Task Parameters ---
        goal_distance_range: Tuple[float, float] = (500.0, 2000.0),
        goal_altitude_range_agl: Tuple[float, float] = (200.0, 1000.0), # AGL
        goal_heading_range_rad: Tuple[float, float] = (0.0, 2 * np.pi), # Radians
        goal_tolerance: float = 50.0,
        reward_type: Literal["Sparse", "Dense"] = "Dense",
        # --- Start Position ---
        start_position: Tuple[float, float, float] = (0.0, 0.0, -500.0), # NED Coords
        # --- Aircraft & Env Params ---
        use_full_aircraft: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 2000,
        time_step: float = 1/60.0,
        normalize_observations: bool = False,
        normalize_actions: bool = False,
    ) -> None:
        """
        Initialize goal environment using structured configuration.

        Args:
            goal_distance_range: (min, max) distance for random goal placement from start (m).
            goal_altitude_range_agl: (min, max) altitude AGL for random goal placement (m).
            goal_heading_range_rad: (min, max) bearing range for random goal placement (radians).
            goal_tolerance: Distance threshold for goal achievement (m).
            reward_type: Reward calculation method ("Sparse" or "Dense").
            start_position: Fixed starting position (x, y, z NED) for the aircraft.
            use_full_aircraft: Whether to use full aircraft model instead of Dubins.
            seed: Random seed for environment.
            render_mode: Type of rendering ("human" or "rgb_array").
            max_episode_steps: Maximum steps per episode before truncation.
            time_step: Simulation time step duration (seconds).
            normalize_observations: Whether to normalize observations globally.
            normalize_actions: Whether to normalize actions globally.
        """

        # --- 1. Setup RNG and Generate Goal ---
        # We need the goal position *before* fully defining the config
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            # Use a default RNG if no seed provided, but seed won't be passed to Rust unless set in EnvConfig
            rng = np.random.RandomState()

        # Generate random goal position based on ranges relative to start_position
        start_pos_vec = np.array(start_position)
        goal_pos_vec = self._generate_goal_position(
            rng=rng,
            origin=start_pos_vec,
            distance_range=goal_distance_range,
            altitude_range_agl=goal_altitude_range_agl,
            heading_range_rad=goal_heading_range_rad
        )
        goal_pos_dataclass = Vec3(x=goal_pos_vec[0], y=goal_pos_vec[1], z=goal_pos_vec[2])

        # --- 2. Build EnvConfig ---
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

        # --- 3. Build Aircraft Physics Config ---
        if use_full_aircraft:
            physics_config = FullAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "full"
        else:
            physics_config = DubinsAircraftPresetBuilder.get_physics_config()
            aircraft_type_str = "dubins"

        # --- 4. Build Start Config (Fixed) ---
        start_config = build_fixed_start_config(position=start_position)

        # --- 5. Build Task Config ---
        task_data = GoalTaskConfigData(
            position=goal_pos_dataclass,
            reward_type=reward_type,
            tolerance=float(goal_tolerance)
        )
        task_config = TaskConfig(type="Goal", config=task_data)

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

        # Store these for potential regeneration on reset if needed (though Rust might handle this)
        self._goal_distance_range = goal_distance_range
        self._goal_altitude_range_agl = goal_altitude_range_agl
        self._goal_heading_range_rad = goal_heading_range_rad
        self._start_position = start_pos_vec


    def _generate_goal_position(
        self,
        rng: np.random.RandomState,
        origin: np.ndarray,
        distance_range: Tuple[float, float],
        altitude_range_agl: Tuple[float, float], # AGL
        heading_range_rad: Tuple[float, float] # Radians
        ) -> np.ndarray:
        """
        Generate a new goal position relative to the origin point.
        """
        distance = rng.uniform(distance_range[0], distance_range[1])
        heading = rng.uniform(heading_range_rad[0], heading_range_rad[1])
        altitude_agl = rng.uniform(altitude_range_agl[0], altitude_range_agl[1])

        # Calculate x,y offset from heading and distance
        x_offset = distance * np.cos(heading)
        y_offset = distance * np.sin(heading)

        # Convert AGL altitude to NED coordinate (negative Down)
        altitude_ned = -altitude_agl

        goal_position = np.array([
            origin[0] + x_offset,
            origin[1] + y_offset,
            altitude_ned
        ])
        return goal_position

    # Override reset if goal needs to be regenerated per episode on the Python side
    # (Alternatively, the Rust backend could handle goal randomization on Reset)
    # def reset( ... ) -> ... :
    #     # 1. Call super().reset() to reset simulation state in Rust
    #     obs, info = super().reset(seed=seed, options=options)
    #     # 2. Generate a *new* goal position here if needed
    #     # 3. Send an *UpdateTask* command (requires defining this command) to Rust
    #     #    with the new goal position.
    #     # 4. Return the initial obs/info from super().reset()
    #     # If Rust handles goal regeneration based on seed, this override isn't needed.
    #     return obs, info

    # REMOVE ALL @classmethod builders (build_sparse_reward, etc.)
