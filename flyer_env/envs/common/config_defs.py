import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Literal, Union

# --- Basic Helper Types ---

@dataclass
class Vec3:
    """Represents a 3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0 # Often represents altitude (negative in NED)

# --- Start Configuration Types ---

@dataclass
class RandomPositionConfig:
    """Configuration for randomizing starting position."""
    origin_x: float = 0.0
    origin_y: float = 0.0
    # Altitudes are typically negative in NED convention (Down)
    min_altitude: float = -600.0 # e.g., 600m AGL
    max_altitude: float = -400.0 # e.g., 400m AGL
    variance: float = 50.0 # Horizontal variance around origin

@dataclass
class RandomSpeedConfig:
    """Configuration for randomizing starting speed."""
    min_speed: float = 75.0 # m/s
    max_speed: float = 100.0 # m/s

@dataclass
class RandomHeadingConfig:
    """Configuration for randomizing starting heading."""
    min_heading: float = 0.0 # Radians
    max_heading: float = 2 * np.pi # Radians

@dataclass
class RandomStartConfigData:
    """Data specific to random start conditions."""
    position: RandomPositionConfig = field(default_factory=RandomPositionConfig)
    speed: RandomSpeedConfig = field(default_factory=RandomSpeedConfig)
    heading: RandomHeadingConfig = field(default_factory=RandomHeadingConfig)
    # Trim condition can be applied even with random start position/heading/speed
    trim_condition: Optional[Dict[str, Any]] = None

@dataclass
class FixedStartConfigData:
    """Data specific to fixed start conditions."""
    position: Vec3 = field(default_factory=Vec3)
    # Optional fixed speed/heading if needed, otherwise defaults might apply
    speed: Optional[float] = None
    heading: Optional[float] = None # Radians
    trim_condition: Optional[Dict[str, Any]] = None # Trim applies here too

@dataclass
class StartConfig:
    """Defines how the aircraft starts."""
    type: Literal["fixed", "random"]
    # Holds either FixedStartConfigData or RandomStartConfigData based on 'type'
    config: Union[FixedStartConfigData, RandomStartConfigData]

# --- Aircraft Physical Configuration ---

@dataclass
class AircraftPhysicsConfig:
    """Configuration holding physical parameters of the aircraft model."""
    # Common parameters (relevant to Dubins and potentially Full)
    max_speed: float = 200.0
    min_speed: float = 15.0
    acceleration: float = 5.0 # Max linear acceleration m/s^2
    max_bank_angle: float = np.radians(45.0) # Max bank angle in radians
    max_turn_rate: float = np.radians(45.0) # Max turn rate in radians/sec
    max_climb_rate: float = 10.0 # Max climb rate m/s
    max_descent_rate: float = 10.0 # Max descent rate m/s

    # Parameters specific to the "Full" aircraft model (if applicable)
    ac_type: Optional[str] = "twin_otter" # Example, could be Literal if types are fixed
    # Add other 'Full' model specific params if they differ from Dubins defaults
    # e.g., mass properties, aerodynamic coefficients file path, etc.

# --- Task Configuration Types ---

@dataclass
class ControlTaskConfigData:
    """Parameters for the 'Control' task."""
    control_type: Literal["Altitude", "Heading", "Speed", "Pitch", "Roll"]
    target: float
    tolerance: float

@dataclass
class GoalTaskConfigData:
    """Parameters for the 'Goal' task."""
    position: Vec3
    reward_type: Literal["Sparse", "Dense"]
    tolerance: float

@dataclass
class LandingTaskConfigData:
    """Parameters for the 'Landing' task."""
    target_position: Optional[Vec3] # None for generic forced landing
    max_landing_speed: float = 25.0
    max_descent_rate: float = 3.0
    max_bank_angle: float = np.radians(30.0) # Radians
    max_landing_distance: float = 200.0 # Lateral distance tolerance at touchdown
    landing_complete_height: float = 0.5 # Height threshold for completion

@dataclass
class RunwayTaskConfigData:
    """Parameters for the 'Runway' task."""
    # These might be fixed or dynamically set by runway_config in EnvConfig
    source: Literal["fixed", "dynamic_runway"] = "dynamic_runway"
    fixed_width: Optional[float] = 45.0
    fixed_length: Optional[float] = 1800.0
    fixed_glideslope: Optional[float] = np.radians(3.0) # Radians
    # Specific position/heading if source is 'fixed'
    fixed_position: Optional[Vec3] = None
    fixed_heading: Optional[float] = None # Radians

@dataclass
class MotionPrimitiveConfig:
    """Base for motion primitives in Trajectory task."""
    type: str # e.g., "StraightAndLevel", "CoordinatedTurn", etc.

@dataclass
class StraightAndLevelMotionConfig(MotionPrimitiveConfig):
    type: Literal["StraightAndLevel"] = "StraightAndLevel"
    target_distance: float = 1000.0

@dataclass
class CoordinatedTurnMotionConfig(MotionPrimitiveConfig):
    type: Literal["CoordinatedTurn"] = "CoordinatedTurn"
    turn_radius: float = 200.0
    turn_angle: float = np.radians(90.0) # Radians
    direction: Literal["Right", "Left"] = "Right"

# TODO: Add ClimbMotionConfig, DescendMotionConfig, StraightAndTurnMotionConfig dataclasses here if needed

@dataclass
class TrajectoryTaskConfigData:
    """Parameters for the 'Trajectory' task."""
    motion: MotionPrimitiveConfig # Holds one of the specific motion configs
    target_velocity: float = 30.0

@dataclass
class TaskConfig:
    """Defines the agent's task."""
    type: Literal["Control", "Goal", "Landing", "Runway", "Trajectory"]
    # Holds one of the specific *TaskConfigData dataclasses
    config: Any

# --- Aircraft Definition (Combines Physics, Start, Task) ---

@dataclass
class AircraftDefinition:
    """Complete definition for a single aircraft in the environment."""
    # --- Fields WITHOUT defaults first ---
    type: Literal["dubins", "full"]
    physics: AircraftPhysicsConfig
    start: StartConfig
    task: TaskConfig
    # --- Fields WITH defaults last ---
    id: str = "agent_0"
    action_type: Literal["Continuous", "Discrete"] = "Continuous"
    observation_type: Literal["Continuous"] = "Continuous"
    normalize_obs: bool = False
    normalize_act: bool = False

# --- Top-Level Environment Configuration ---

@dataclass
class AgentRenderConfig:
     """Configuration related to rendering from the agent's perspective."""
     render_width: float = 800.0
     render_height: float = 600.0
     mode: Literal["human", "RGBArray", "Headless"] = "human"

@dataclass
class RunwayGenerationConfig:
    """Configuration for how the runway is generated (for RunwayEnv)."""
    generation_type: Literal["fixed", "random_relative_to_aircraft_start"] = "random_relative_to_aircraft_start"
    # Params for random generation
    distance_range: Tuple[float, float] = (4000.0, 8000.0) # meters
    bearing_range: Tuple[float, float] = (0.0, 2 * np.pi) # radians
    heading_range: Tuple[float, float] = (0.0, 2 * np.pi) # radians
    # Fixed geometry used by both generation types
    fixed_width: float = 45.0
    fixed_length: float = 1800.0
    # Fixed position/heading if generation_type is 'fixed'
    fixed_position: Optional[Vec3] = None
    fixed_heading: Optional[float] = None # Radians

@dataclass
class EnvConfig:
    """Base configuration shared across environment types."""
    seed: Optional[int] = None
    max_episode_steps: int = 1000
    time_step: float = 1/60.0 # Simulation time step (s)
    steps_per_action: int = 4 # How many simulation steps per agent action
    normalize_observations: bool = False # Global flag for observation normalization
    normalize_actions: bool = False # Global flag for action normalization
    agent_config: AgentRenderConfig = field(default_factory=AgentRenderConfig)
    # Specific configs for env types that need them
    runway_config: Optional[RunwayGenerationConfig] = None
    # Could add terrain_config, physics_config (gravity, atmosphere), etc. here

@dataclass
class SingleAgentEnvConfig:
    """Complete configuration object passed to SingleAgentEnv."""
    env: EnvConfig
    aircraft: AircraftDefinition # Exactly one aircraft definition
