from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
from flyer_env.envs.common.single_agent_env import SingleAgentEnv

class DubinsAircraftPreset:
    @staticmethod
    def default() -> Dict[str, Any]:
        return {
            "type": "dubins",
            "config": {
                "max_speed": 200.0,
                "min_speed": 15.0,
                "acceleration": 5.0,
                "max_bank_angle": 45.0,  # degrees
                "max_turn_rate": 45.0,   # degrees/s
                "max_climb_rate": 10.0,
                "max_descent_rate": 10.0
            }
        }

    @staticmethod
    def get_start_config(
        initial_altitude: float = 500.0,
        initial_heading: Optional[float] = None,
        initial_speed: Optional[float] = None,
        position_variance: float = 50.0,
    ) -> Dict[str, Any]:
        config = {
            "type": "random",
            "config": {
                "position": {
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                    "min_altitude": initial_altitude,
                    "max_altitude": initial_altitude,
                    "variance": position_variance
                },
                "speed": {
                    "min_speed": 75.0 if initial_speed is None else initial_speed,
                    "max_speed": 100.0 if initial_speed is None else initial_speed
                },
                "heading": {
                    "min_heading": 0.0 if initial_heading is None else initial_heading,
                    "max_heading": 2 * np.pi if initial_heading is None else initial_heading
                }
            }
        }
        return config

class FullAircraftPreset:
    @staticmethod
    def default() -> Dict[str, Any]:
        return {
            "type": "full",
            "config": {
                "ac_type": "generic_transport"
            }
        }

    @staticmethod
    def get_start_config(
        initial_altitude: float = 1000.0,
        initial_heading: Optional[float] = None,
        initial_speed: float = 50.0,
        position_variance: float = 50.0,
    ) -> Dict[str, Any]:
        config = {
            "type": "random",
            "config": {
                "position": {
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                    "min_altitude": initial_altitude,
                    "max_altitude": initial_altitude,
                    "variance": position_variance
                },
                "speed": {
                    "min_speed": initial_speed,
                    "max_speed": initial_speed
                },
                "heading": {
                    "min_heading": 0.0 if initial_heading is None else initial_heading,
                    "max_heading": 2 * np.pi if initial_heading is None else initial_heading
                }
            }
        }
        return config

class ControlFlyerEnv(SingleAgentEnv):
    """
    Environment for fundamental flight parameter regulation tasks.
    Focuses on precise control of individual state variables.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        control_type: str = "altitude",
        target_value: float = 1000.0,
        tolerance: float = 10.0,
        start_deviation: Union[float, Tuple[float, float]] = 100.0,
        use_full_aircraft: bool = False,
        env_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize control environment with specific parameters.

        Args:
            seed: Random seed for environment
            render_mode: Type of rendering ("human" or "rgb_array")
            control_type: Type of control task ("altitude", "heading", "speed", "pitch", "roll")
            target_value: Target value to maintain
            tolerance: Acceptable deviation from target
            start_deviation: How far from target to start:
                           - float: symmetric deviation [-value, value]
                           - tuple: (min_deviation, max_deviation)
            use_full_aircraft: Whether to use full aircraft model instead of Dubins
            env_config: Optional additional environment configuration
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        if env_config is None:
            env_config = {}

        env_config.setdefault("max_episode_steps", 1000)
        env_config.setdefault("time_step", 1/60)

        if seed is not None:
            env_config["seed"] = seed

        # Handle start deviations
        if isinstance(start_deviation, (int, float)):
            min_dev, max_dev = -abs(start_deviation), abs(start_deviation)
        else:
            min_dev, max_dev = start_deviation

        # Prepare aircraft configuration
        if use_full_aircraft:
            aircraft_preset = FullAircraftPreset()
        else:
            aircraft_preset = DubinsAircraftPreset()

        # Configure starting conditions based on control type
        initial_altitude = target_value + min_dev if control_type == "altitude" else 500.0
        initial_heading = (target_value + min_dev) if control_type == "heading" else None
        initial_speed = (target_value + min_dev) if control_type == "speed" else None

        aircraft_config = aircraft_preset.default()
        aircraft_config["start_config"] = aircraft_preset.get_start_config(
            initial_altitude=initial_altitude,
            initial_heading=initial_heading,
            initial_speed=initial_speed
        )

        # Create task configuration
        task_config = {
            "type": "Control",
            "config": {
                "control_type": control_type.capitalize(),
                "target": float(target_value),
                "tolerance": float(tolerance)
            }
        }

        # Combine configurations
        full_aircraft_config = [{
            **aircraft_config,
            "action_type": "Continuous",
            "observation_type": "Continuous",
            "task_config": task_config
        }]

        env_config["aircraft_config"] = full_aircraft_config

        super().__init__(config=env_config, render_mode=render_mode)

    @classmethod
    def build_altitude_control(
        cls,
        target_altitude: float = 500.0,
        tolerance: float = 10.0,
        start_deviation: Union[float, Tuple[float, float]] = 100.0,
        use_full_aircraft: bool = False,
        **kwargs
    ) -> "ControlFlyerEnv":
        """Create environment for altitude control task."""
        return cls(
            control_type="altitude",
            target_value=target_altitude,
            tolerance=tolerance,
            start_deviation=start_deviation,
            use_full_aircraft=use_full_aircraft,
            **kwargs
        )

    @classmethod
    def build_heading_control(
        cls,
        target_heading: float = 0.0,
        tolerance: float = 0.1,
        start_deviation: Union[float, Tuple[float, float]] = 0.5,
        use_full_aircraft: bool = False,
        **kwargs
    ) -> "ControlFlyerEnv":
        """Create environment for heading control task."""
        return cls(
            control_type="heading",
            target_value=target_heading,
            tolerance=tolerance,
            start_deviation=start_deviation,
            use_full_aircraft=use_full_aircraft,
            **kwargs
        )

    @classmethod
    def build_speed_control(
        cls,
        target_speed: float = 30.0,
        tolerance: float = 2.0,
        start_deviation: Union[float, Tuple[float, float]] = 5.0,
        use_full_aircraft: bool = False,
        **kwargs
    ) -> "ControlFlyerEnv":
        """Create environment for speed control task."""
        return cls(
            control_type="speed",
            target_value=target_speed,
            tolerance=tolerance,
            start_deviation=start_deviation,
            use_full_aircraft=use_full_aircraft,
            **kwargs
        )

    @classmethod
    def build_attitude_control(
        cls,
        attitude_type: str = "pitch",
        target_angle: float = 0.0,
        tolerance: float = 0.05,
        start_deviation: Union[float, Tuple[float, float]] = 0.2,
        **kwargs
    ) -> "ControlFlyerEnv":
        """Create environment for attitude control task."""
        if attitude_type not in ["pitch", "roll"]:
            raise ValueError("attitude_type must be 'pitch' or 'roll'")

        return cls(
            control_type=attitude_type,
            target_value=target_angle,
            tolerance=tolerance,
            start_deviation=start_deviation,
            use_full_aircraft=True,  # Attitude control always requires full aircraft
            **kwargs
        )
