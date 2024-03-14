import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Callable, Dict
from gymnasium import spaces
import numpy as np

from pyflyer import Aircraft
from flyer_env import utils
from flyer_env.utils import Vector
from flyer_env.aircraft import ControlledAircraft

if TYPE_CHECKING:
    from flyer_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType:

    def __init__(self, env: "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space"""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """The class of vehicle capable of executing this action"""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """Execute the action on the ego vehicle"""
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon"""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):

    ELEVATOR_RANGE = (-5.0 * (np.pi / 180.0), 30.0 * (np.pi / 180.0))
    AILERON_RANGE = (-5.0 * (np.pi / 180.0), 5.0 * (-np.pi / 180.0))
    TLA_RANGE = (0.0, 1.0)
    RUDDER_RANGE = (
        -30.0 * (np.pi / 180.0),
        30.0 * (np.pi / 180.0),
    )  # removed rudder from action-space for now

    """
    A continuous action space for thrust-lever-angle and control surface deflections.
    Controls are set in order [elevator, aileron, tla, rudder].
    """

    def __init__(
        self,
        env: "AbstractEnv",
        elevator_range: Optional[Tuple[float, float]] = None,
        aileron_range: Optional[Tuple[float, float]] = None,
        tla_range: Optional[Tuple[float, float]] = None,
        powered: bool = True,
        clip: bool = True,
        **kwargs
    ) -> None:
        """
        Create a continuous action space
        """
        super().__init__(env)

        # Setup control limit ranges
        self.elevator_range = elevator_range if elevator_range else self.ELEVATOR_RANGE
        self.aileron_range = aileron_range if aileron_range else self.AILERON_RANGE
        self.tla_range = tla_range if tla_range else self.TLA_RANGE

        self.powered = powered
        self.clip = clip
        self.size = 3 if self.powered else 2

        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Aircraft

    def act(self, action: np.ndarray) -> None:
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.powered:
            self.controlled_vehicle.act(
                {
                    "aileron": utils.lmap(action[0], [-1, 1], self.aileron_range),
                    "elevator": utils.lmap(action[1], [-1, 1], self.elevator_range),
                    "tla": utils.lmap(action[2], [-1, 1], self.tla_range),
                    # 'rudder': utils.lmap(action[3], [-1, 1], self.rudder_range)
                    "rudder": 0.0,
                }
            )
        else:
            self.controlled_vehicle.act(
                {
                    "aileron": utils.lmap(action[0], [-1, 1], self.aileron_range),
                    "elevator": utils.lmap(action[1], [-1, 1], self.elevator_range),
                    "tla": 0.0,
                    # 'rudder': utils.lmap(action[2], [-1, 1], self.rudder_range)
                    "rudder": 0.0,
                }
            )
        self.last_action = action


class LongitudinalAction(ActionType):

    ELEVATOR_RANGE = (-5.0 * (np.pi / 180.0), 30.0 * (np.pi / 180.0))
    TLA_RANGE = (0.0, 1.0)

    """
    A continuous action space for thrust-lever-angle and elevator control.
    Controls are set in order [elevator, tla].
    """

    def __init__(
        self,
        env: "AbstractEnv",
        elevator_range: Optional[Tuple[float, float]] = None,
        tla_range: Optional[Tuple[float, float]] = None,
        powered: bool = True,
        clip: bool = True,
        **kwargs
    ) -> None:
        """
        Create a continuous longitudinally constrained action space
        """
        super().__init__(env)

        # Setup control limit ranges
        self.elevator_range = elevator_range if elevator_range else self.ELEVATOR_RANGE
        self.tla_range = tla_range if tla_range else self.TLA_RANGE

        self.powered = powered
        self.clip = clip
        self.size = 2 if self.powered else 1

        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Aircraft

    def act(self, action: np.ndarray) -> None:
        if self.clip:
            action = np.clip(action, -1.0, 1.0)
        if self.powered:
            self.controlled_vehicle.act(
                {
                    "aileron": 0.0,
                    "elevator": utils.lmap(action[0], [-1.0, 1.0], self.elevator_range),
                    "tla": utils.lmap(action[1], [-1, 1], self.tla_range),
                    "rudder": 0.0,
                }
            )
        else:
            self.controlled_vehicle.act(
                {
                    "aileron": 0.0,
                    "elevator": utils.lmap(action[0], [-1.0, 1.0], self.elevator_range),
                    "tla": 0.0,
                    "rudder": 0.0,
                }
            )
        self.last_action = action


class HeadingAction(ActionType):

    HEADING_RANGE = (-np.pi, np.pi)

    """
    A continuous action space with a fixed altitude and speed.
    Controls are only [aileron].
    """

    def __init__(
        self,
        env: "AbstractEnv",
        heading_range: Optional[Tuple[float, float]] = None,
        powered: bool = True,
        clip: bool = True,
        **kwargs
    ) -> None:
        """
        Create a continuous laterally constrained action space
        """
        super().__init__(env)

        self.heading_range = heading_range if heading_range else self.HEADING_RANGE
        self.powered = powered
        self.clip = clip
        self.size = 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(
            self.heading_range[0],
            self.heading_range[1],
            shape=(self.size,),
            dtype=np.float32,
        )

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(ControlledAircraft)

    def act(self, action: np.ndarray) -> None:
        """
        Apply the action to the controlled vehicle

        :param action: action array with [sine, cosine] mapped between ranges
        """

        if self.clip:
            action = np.clip(action, self.heading_range[0], self.heading_range[1])

        if self.powered:
            self.controlled_vehicle.act(
                {"heading": action, "alt": -1000.0, "speed": 80.0}
            )
        self.last_action = action


class ControlledAction(ActionType):
    """
    An action that controls the aircraft using a PID controller to track towards the target.
    Controls are set in the order: [HeadingSin, HeadingCos, Alt, Speed]
    """

    HEADING_RANGE = (-np.pi, np.pi)
    ALT_RANGE = (0.0, -10000)
    SPEED_RANGE = (60.0, 110.0)

    def __init__(
        self,
        env: "AbstractEnv",
        heading_range: Optional[Tuple[float, float]] = None,
        alt_range: Optional[Tuple[float, float]] = None,
        speed_range: Optional[Tuple[float, float]] = None,
        clip: bool = True,
        **kwargs
    ) -> None:
        """
        Create a controlled action space

        :param env: the environment
        :param heading_range: the range of heading values [rads]
        :param alt_range: the range of altitude values [m]
        :param speed_range: the range of speed values [m/s]
        :param clip: clip the action to the desired range
        """
        super().__init__(env)
        self.heading_range = heading_range if heading_range else self.HEADING_RANGE
        self.alt_range = alt_range if alt_range else self.ALT_RANGE
        self.speed_range = speed_range if speed_range else self.SPEED_RANGE
        self.size = 4  # pass heading as sine and cosine components
        self.clip = clip
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(ControlledAircraft)

    def act(self, action: np.array) -> None:
        """
        Apply the action to the controlled vehicle

        :param action: action array with [sine, cosine, altitude and speed] mapped between ranges
        """
        if self.clip:
            action = np.clip(action, -1, 1)

        self.controlled_vehicle.act(
            {
                "heading": np.arctan2(action[0], action[1]),
                "alt": utils.lmap(action[2], [-1, 1], self.alt_range),
                "speed": utils.lmap(action[3], [-1, 1], self.speed_range),
            }
        )
        self.last_action = action


class PursuitAction(ActionType):

    ALT_RANGE = (0.0, 10000.0)
    SPEED_RANGE = (0.0, 300.0)

    """
    A high level action to navigate to a 2D point in space at a fixed altitude and speed.
    Actions are set in the order: [{GoalPosX, GoalPosY}, {Alt, Speed}]
    """

    def __init__(
        self,
        env: "AbstractEnv",
        alt_range: Optional[Tuple[float, float]] = None,
        speed_range: Optional[Tuple[float, float]] = None,
        clip: bool = True,
        **kwargs
    ) -> None:

        super().__init__(env)
        self.alt_range = alt_range if alt_range else self.ALT_RANGE
        self.speed_range = speed_range if speed_range else self.SPEED_RANGE
        self.size = 4
        self.clip = clip
        self.last_action = {"goal_pos": np.zeros(2), "other_controls": np.zeros(2)}

    def space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "goal_pos": spaces.Box(
                    low=-np.infty, high=np.infty, shape=(2,), dtype=np.float32
                ),
                "other_controls": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(ControlledAircraft)

    def act(self, action: Dict[str, Vector]) -> None:
        """
        Apply the action to the controlled vehicle

        :param action: dictionary corresponding to unbounded GoalPos and bounded Alt and Speed
        """

        if self.clip:
            action["other_controls"] = np.clip(action["other_controls"], -1, 1)

        self.controlled_vehicle.act(
            {
                "pursuit_target": action["goal_pos"],
                "alt": utils.lmap(action["other_controls"][0], [0, 1], self.alt_range),
                "speed": utils.lmap(
                    action["other_controls"][1], [0, 1], self.speed_range
                ),
            }
        )

        self.last_action = action


class TrackAction(ActionType):

    SPEED_RANGE = (0.0, 300.0)

    def __init__(
        self,
        env: "AbstractEnv",
        speed_range: Optional[Tuple[float, float]] = None,
        clip: bool = None,
        **kwargs
    ) -> None:

        super().__init__(env)
        self.speed_range = speed_range if speed_range else self.SPEED_RANGE
        self.size = 4
        self.clip = clip
        self.last_action = {"track_points": np.zeros(3), "other_controls": np.zeros(1)}

    def space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "targets": spaces.Box(
                    low=-np.infty, high=np.infty, shape=(3,), dtype=np.float32
                ),
                "other_controls": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(ControlledAircraft)

    def act(self, action: Dict[str, Vector]) -> None:
        """
        Apply the action to the controlled vehicle

        :param action: dictionary corresponding to commanded positions, altitude and speed
        """

        if self.clip:
            action["other_controls"] = np.clip(action["other_controls"], -1, 1)

        self.controlled_vehicle.act(
            {
                "track_points": action["targets"],
                "speed": utils.lmap(action["other_controls"], [0, 1], self.speed_range),
            }
        )

        self.last_action = action


def action_factory(env: "AbstractEnv", config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "LongitudinalAction":
        return LongitudinalAction(env, **config)
    elif config["type"] == "HeadingAction":
        return HeadingAction(env, **config)
    elif config["type"] == "ControlledAction":
        return ControlledAction(env, **config)
    elif config["type"] == "PursuitAction":
        return PursuitAction(env, **config)
    elif config["type"] == "TrackAction":
        return TrackAction(env, **config)
    else:
        raise ValueError("Unknown action type")
