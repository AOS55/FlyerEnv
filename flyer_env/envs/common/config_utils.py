import numpy as np
from typing import Dict, Any, Optional, Tuple, Literal

# Import the dataclasses defined in Step 1
from flyer_env.envs.common.config_defs import (
    AircraftPhysicsConfig,
    StartConfig,
    RandomStartConfigData,
    RandomPositionConfig,
    RandomSpeedConfig,
    RandomHeadingConfig,
    FixedStartConfigData, # If needed for other builders in the future
    Vec3,
    # Import specific task configs if you add task builders here
    ControlTaskConfigData, TaskConfig
)

# --- Aircraft Physics Configuration Builders ---

class AircraftPresetBuilder:
    """Base class for aircraft preset builders (optional, for structure)."""
    @staticmethod
    def get_physics_config() -> AircraftPhysicsConfig:
        raise NotImplementedError

class DubinsAircraftPresetBuilder(AircraftPresetBuilder):
    """Builds configurations related to Dubins aircraft."""

    @staticmethod
    def get_physics_config() -> AircraftPhysicsConfig:
        """Returns default physics parameters for a Dubins aircraft."""
        # Return the dataclass instance
        return AircraftPhysicsConfig(
            max_speed=200.0,
            min_speed=15.0,
            acceleration=5.0,
            max_bank_angle=np.radians(45.0), # Convert to radians
            max_turn_rate=np.radians(45.0),  # Convert to radians
            max_climb_rate=10.0,
            max_descent_rate=10.0,
            ac_type=None # Dubins doesn't have a specific type like 'twin_otter'
        )

class FullAircraftPresetBuilder(AircraftPresetBuilder):
    """Builds configurations related to the Full physics aircraft model."""

    @staticmethod
    def get_physics_config(ac_type: str = "twin_otter") -> AircraftPhysicsConfig:
        """
        Returns default physics parameters for a Full aircraft model.
        Note: Specific detailed physics are usually loaded from aircraft data files
              within the Rust backend based on ac_type. This config primarily holds
              limits or overrides if needed by the Python side or factories.
        """
        # Return the dataclass instance
        # These defaults might be less meaningful if Rust handles everything.
        return AircraftPhysicsConfig(
            max_speed=250.0,  # Example placeholder
            min_speed=30.0,   # Example placeholder
            acceleration=8.0, # Example placeholder
            max_bank_angle=np.radians(60.0), # Example placeholder
            max_turn_rate=np.radians(50.0), # Example placeholder
            max_climb_rate=15.0, # Example placeholder
            max_descent_rate=15.0, # Example placeholder
            ac_type=ac_type # Store the aircraft type string
        )

    @staticmethod
    def build_trim_condition(
        trim_type: Literal["StraightAndLevel", "SteadyClimb", "CoordinatedTurn"],
        airspeed: float,
        gamma_deg: Optional[float] = None, # Flight path angle (degrees)
        bank_angle_deg: Optional[float] = None # Bank angle (degrees)
        ) -> Dict[str, Any]:
        """
        Builds the dictionary representing a trim condition request for the Rust backend.
        """
        trim_config = {
            "condition_type": trim_type,
            "airspeed": float(airspeed)
        }
        if trim_type == "SteadyClimb" and gamma_deg is not None:
            trim_config["gamma"] = float(np.radians(gamma_deg)) # Convert to radians
        if trim_type == "CoordinatedTurn" and bank_angle_deg is not None:
            trim_config["bank_angle"] = float(np.radians(bank_angle_deg)) # Convert to radians
        return trim_config


# --- Start Configuration Builders ---

def build_random_start_config(
    min_altitude_agl: float = 400.0,
    max_altitude_agl: float = 600.0,
    min_speed: Optional[float] = 75.0,
    max_speed: Optional[float] = 100.0,
    min_heading_deg: Optional[float] = 0.0, # Degrees
    max_heading_deg: Optional[float] = 360.0, # Degrees
    position_origin_x: float = 0.0,
    position_origin_y: float = 0.0,
    position_variance: float = 50.0,
    trim_condition: Optional[Dict[str, Any]] = None,
) -> StartConfig:
    """
    Builds a StartConfig dataclass for randomized starting conditions.
    """
    # Convert AGL altitude range to NED (negative Down) coordinates
    min_alt_ned = -max_altitude_agl
    max_alt_ned = -min_altitude_agl

    # Convert heading range from degrees to radians
    min_heading_rad = np.radians(min_heading_deg) if min_heading_deg is not None else 0.0
    max_heading_rad = np.radians(max_heading_deg) if max_heading_deg is not None else 2 * np.pi

    random_pos_cfg = RandomPositionConfig(
        origin_x=float(position_origin_x),
        origin_y=float(position_origin_y),
        min_altitude=float(min_alt_ned),
        max_altitude=float(max_alt_ned),
        variance=float(position_variance)
    )

    random_speed_cfg = RandomSpeedConfig(
        min_speed=float(min_speed),
        max_speed=float(max_speed)
    )

    random_heading_cfg = RandomHeadingConfig(
        min_heading=float(min_heading_rad),
        max_heading=float(max_heading_rad)
    )

    random_start_data = RandomStartConfigData(
        position=random_pos_cfg,
        speed=random_speed_cfg,
        heading=random_heading_cfg,
        trim_condition=trim_condition # Pass trim dict directly
    )
    # Return the StartConfig dataclass instance
    return StartConfig(type="random", config=random_start_data)


def build_fixed_start_config(
    position: Tuple[float, float, float],
    speed: Optional[float] = None,
    heading_deg: Optional[float] = None, # Degrees
    trim_condition: Optional[Dict[str, Any]] = None,
) -> StartConfig:
    """
    Builds a StartConfig dataclass for fixed starting conditions.
    """
    start_pos = Vec3(x=float(position[0]), y=float(position[1]), z=float(position[2]))
    heading_rad = np.radians(heading_deg) if heading_deg is not None else None

    fixed_start_data = FixedStartConfigData(
        position=start_pos,
        speed=float(speed) if speed is not None else None,
        heading=float(heading_rad) if heading_rad is not None else None,
        trim_condition=trim_condition
    )
    # Return the StartConfig dataclass instance
    return StartConfig(type="fixed", config=fixed_start_data)


# --- Task Configuration Builders (Example for Control Task) ---

def build_control_task_config(
    control_type: Literal["Altitude", "Heading", "Speed", "Pitch", "Roll"],
    target_value: float,
    tolerance: float
) -> TaskConfig:
    """Builds a TaskConfig for a control task."""
    task_data = ControlTaskConfigData(
        control_type=control_type,
        target=float(target_value),
        tolerance=float(tolerance)
    )
    # Return the TaskConfig dataclass instance
    return TaskConfig(type="Control", config=task_data)

# --- Add similar builder functions for Goal, Landing, Runway, Trajectory tasks here ---
# Example:
# def build_goal_task_config(...) -> TaskConfig: ...
# def build_landing_task_config(...) -> TaskConfig: ...
