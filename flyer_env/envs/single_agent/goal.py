from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from flyer_env.envs.common.single_agent_env import SingleAgentEnv

class GoalFlyerEnv(SingleAgentEnv):
    """Environment for 3D waypoint navigation tasks."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        distance_range: Tuple[float, float] = (500.0, 2000.0),
        altitude_range: Tuple[float, float] = (-1000.0, -200.0),
        heading_range: Tuple[float, float] = (0.0, 2 * np.pi),
        origin: Tuple[float, float, float] = (0.0, 0.0, -500.0),
        tolerance: float = 50.0,
        reward_type: str = "Dense",
        use_full_aircraft: bool = False,
        episode_length: Optional[int] = 1000,
        env_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize goal environment.

        Args:
            seed: Random seed for environment
            render_mode: Type of rendering ("human" or "rgb_array")
            distance_range: (min_distance, max_distance) from origin in meters
            altitude_range: (min_altitude, max_altitude) in meters
            heading_range: (min_heading, max_heading) in radians for goal distribution
            origin: (x, y, z) coordinates for goal distribution reference
            tolerance: Distance threshold for goal achievement
            reward_type: Reward calculation method ("sparse" or "dense")
            use_full_aircraft: Whether to use full aircraft model instead of Dubins
            env_config: Optional additional configuration
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        if env_config is None:
            env_config = {}

        env_config.setdefault("max_episode_steps", 2000)
        env_config.setdefault("time_step", 1/60)

        if seed is not None:
            rng = np.random.RandomState(seed)
            env_config["seed"] = seed
        else:
            rng = np.random.RandomState()

        # Store goal generation parameters
        self.distance_range = distance_range
        self.altitude_range = altitude_range
        self.heading_range = heading_range
        self.origin = np.array(origin)

        # Generate initial goal position using seeded RNG
        goal_position = self._generate_goal_position(rng)

        # Configure aircraft
        aircraft_type = "full" if use_full_aircraft else "dubins"
        aircraft_config = [{
            "type": aircraft_type,
            "action_type": "Continuous",
            "observation_type": "Continuous",
            "start_config": {
                "type": "fixed",
                "config": {
                    "position": {"x": origin[0], "y": origin[1], "z": origin[2]},  # Start at origin
                }
            },
            "task_config": {
                "type": "Goal",
                "config": {
                    "position": {"x": goal_position[0], "y": goal_position[1], "z": goal_position[2]},
                    "reward_type": reward_type,
                    "tolerance": float(tolerance)
                }
            }
        }]

        env_config["aircraft_config"] = aircraft_config

        super().__init__(config=env_config, render_mode=render_mode)

    def _generate_goal_position(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate a new goal position relative to the origin point.

        Args:
            rng: Random number generator to use

        Returns:
            3D goal position [x, y, z]
        """
        # Sample random distance and heading
        distance = rng.uniform(self.distance_range[0], self.distance_range[1])
        heading = rng.uniform(self.heading_range[0], self.heading_range[1])

        # Sample altitude within range
        altitude = rng.uniform(self.altitude_range[0], self.altitude_range[1])

        # Calculate x,y offset from heading and distance
        x_offset = distance * np.cos(heading)
        y_offset = distance * np.sin(heading)

        # Add offset to origin position
        goal_position = np.array([
            self.origin[0] + x_offset,
            self.origin[1] + y_offset,
            altitude  # Use absolute altitude from range
        ])

        return goal_position

    @classmethod
    def build_sparse_reward(
        cls,
        distance_range: Tuple[float, float] = (500.0, 2000.0),
        altitude_range: Tuple[float, float] = (-1000.0, -200.0),
        heading_range: Tuple[float, float] = (0.0, 2 * np.pi),
        origin: Tuple[float, float, float] = (0.0, 0.0, -500.0),
        tolerance: float = 50.0,
        use_full_aircraft: bool = False,
        **kwargs
    ) -> "GoalFlyerEnv":
        """Create environment with sparse rewards."""
        return cls(
            distance_range=distance_range,
            altitude_range=altitude_range,
            heading_range=heading_range,
            origin=origin,
            tolerance=tolerance,
            reward_type="sparse",
            use_full_aircraft=use_full_aircraft,
            **kwargs
        )

    @classmethod
    def build_dense_reward(
        cls,
        distance_range: Tuple[float, float] = (500.0, 2000.0),
        altitude_range: Tuple[float, float] = (-1000.0, -200.0),
        heading_range: Tuple[float, float] = (0.0, 2 * np.pi),
        origin: Tuple[float, float, float] = (0.0, 0.0, -500.0),
        tolerance: float = 50.0,
        use_full_aircraft: bool = False,
        **kwargs
    ) -> "GoalFlyerEnv":
        """Create environment with dense distance-based rewards."""
        return cls(
            distance_range=distance_range,
            altitude_range=altitude_range,
            heading_range=heading_range,
            origin=origin,
            tolerance=tolerance,
            reward_type="dense",
            use_full_aircraft=use_full_aircraft,
            **kwargs
        )
