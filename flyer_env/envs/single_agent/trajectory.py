from typing import Dict, Any, Optional, List
import numpy as np
from flyer_env.envs.common.single_agent_env import SingleAgentEnv

class TrajectoryFlyerEnv(SingleAgentEnv):
    """Environment for executing specific motion primitives."""

    def __init__(
        self,
        motion_type: str = "straight_and_level",
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None,
        **motion_params
    ) -> None:
        """
        Initialize trajectory environment.

        Args:
            motion_type: Type of motion primitive
            target_velocity: Desired airspeed in m/s
            motion_params: Parameters specific to motion type
            config: Optional additional configuration
        """
        if config is None:
            config = {}

        config.setdefault("max_episode_steps", 1000)
        config.setdefault("time_step", 1/60)

        # Configure motion primitive based on type
        motion_config = self._build_motion_config(motion_type, motion_params)

        # Create task configuration
        task_config = {
            "type": "Trajectory",
            "config": {
                "motion_type": motion_config,
                "target_velocity": float(target_velocity)
            }
        }

        # Configure aircraft
        aircraft_config = [{
            "type": "dubins",  # Trajectory tasks use Dubins aircraft
            "action_type": "Continuous",
            "observation_type": "Continuous",
            "task_config": task_config
        }]

        config["aircraft_config"] = aircraft_config

        super().__init__(config=config)

    def _build_motion_config(self, motion_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build motion primitive configuration."""
        if motion_type == "straight_and_level":
            return {
                "type": "StraightAndLevel",
                "target_distance": float(params.get("target_distance", 1000.0))
            }
        elif motion_type == "coordinated_turn":
            return {
                "type": "CoordinatedTurn",
                "turn_radius": float(params.get("turn_radius", 200.0)),
                "turn_angle": float(params.get("turn_angle", 90.0)),
                "direction": params.get("direction", "Right")
            }
        elif motion_type == "climb":
            return {
                "type": "Climb",
                "target_climb_rate": float(params.get("target_climb_rate", 5.0))
            }
        elif motion_type == "descend":
            return {
                "type": "Descend",
                "target_descent_rate": float(params.get("target_descent_rate", 5.0))
            }
        elif motion_type == "straight_and_turn":
            return {
                "type": "StraightAndTurn",
                "straight_distance": float(params.get("straight_distance", 500.0)),
                "turn_radius": float(params.get("turn_radius", 200.0)),
                "turn_angle": float(params.get("turn_angle", 90.0)),
                "direction": params.get("direction", "Right")
            }
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")

    @classmethod
    def build_straight_and_level(
        cls,
        target_distance: float = 1000.0,
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryFlyerEnv":
        """Create environment for straight and level flight."""
        return cls("straight_and_level", target_velocity, config, target_distance=target_distance)

    @classmethod
    def build_coordinated_turn(
        cls,
        turn_radius: float = 200.0,
        turn_angle: float = 90.0,
        direction: str = "Right",
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryFlyerEnv":
        """Create environment for coordinated turn maneuver."""
        return cls("coordinated_turn", target_velocity, config,
                  turn_radius=turn_radius, turn_angle=turn_angle, direction=direction)

    @classmethod
    def build_climb(
        cls,
        target_climb_rate: float = 5.0,
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryFlyerEnv":
        """Create environment for climbing maneuver."""
        return cls("climb", target_velocity, config, target_climb_rate=target_climb_rate)

    @classmethod
    def build_descend(
        cls,
        target_descent_rate: float = 5.0,
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryFlyerEnv":
        """Create environment for descent maneuver."""
        return cls("descend", target_velocity, config, target_descent_rate=target_descent_rate)

    @classmethod
    def build_straight_and_turn(
        cls,
        straight_distance: float = 500.0,
        turn_radius: float = 200.0,
        turn_angle: float = 90.0,
        direction: str = "Right",
        target_velocity: float = 30.0,
        config: Optional[Dict[str, Any]] = None
    ) -> "TrajectoryFlyerEnv":
        """Create environment for straight flight followed by turn."""
        return cls("straight_and_turn", target_velocity, config,
                  straight_distance=straight_distance, turn_radius=turn_radius,
                  turn_angle=turn_angle, direction=direction)
