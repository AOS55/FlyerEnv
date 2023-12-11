import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Dict, TYPE_CHECKING, Tuple, OrderedDict

if TYPE_CHECKING:
    from flyer_env.envs.common.abstract import AbstractEnv


class ObservationType:
    
    def __init__(self, env: "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None
    
    def space(self) -> spaces.Space:
        """Get the observation space"""
        raise NotImplementedError
    
    def observe(self): 
        """Get an observation of the environment state"""
        raise NotImplementedError
    
    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene

        If not set, the first controlled vehicle is used.
        """
        return self.__observer_vehicle or self.env.vehicle
    
    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class DynamicObservation(ObservationType):
    
    """Observe the dynamics of a vehicle"""

    FEATURES: List[str] = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 vehicles_count: int = 1,
                 features_range: Dict[str, List[float]] = None,
                 **kwargs: dict) -> None:

        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)
    
    def observe(self) -> np.ndarray:

        # TODO: this is probably a slow way to collect data, can we speed it up?
        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        return obs.astype(self.space().dtype)


class TrajectoryObservation(ObservationType):

    """
    Observe dynamics of vehicle relative to goal position
    ONLY FOR USE WITH TRAJECTORY ENV
    """

    FEATURES: List[str] = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 vehicles_count: int = 1,
                 features_range: Dict[str, List[float]] = None,
                 **kwargs: dict) -> None:
        
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        if hasattr(env, "goal"):
            self.goal = env.goal

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)
    
    def observe(self) -> np.ndarray:

        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        obs[0, 0] = self.goal[0] - obs[0, 0]
        obs[0, 1] = self.goal[1] - obs[0, 1]
        obs[0, 2] = self.goal[2] - obs[0, 2]
        return obs.astype(self.space().dtype)

class LateralTrajectoryObservation(ObservationType):

    """
    Observe dynamics of vehicle relative to goal position, restricted to horizontal plane
    ONLY FOR USE WITH TRAJECTORY ENV
    """

    FEATURES: List[str] = ['x', 'y', 'u', 'v', 'yaw']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 vehicles_count: int = 1,
                 features_range: Dict[str, List[float]] = None,
                 **kwargs: dict) -> None:
        
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        if hasattr(env, "goal"):
            self.goal = env.goal

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)
    
    def observe(self) -> np.ndarray:

        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        obs[0, 0] = self.goal[0] - obs[0, 0]
        obs[0, 1] = self.goal[1] - obs[0, 1]
        return obs.astype(self.space().dtype)

class ControlObservation(ObservationType):
    
    """
    Observe the aircaft without the position information
    """

    FEATURES: List[str] = ['roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 vehicles_count: int = 1,
                 features_range: Dict[str, List[float]] = None,
                 **kwargs: dict) -> None:
        
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)
    
    def observe(self) -> np.ndarray:

        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        return obs.astype(self.space().dtype)


class LongitudinalObservation(ObservationType):

    """
    Observe the aircraft only given longitudinal data
    """

    FEATURES: List[str] = ['pitch', 'u', 'w', 'q']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 vehicles_count: int = 1,
                 features_range: Dict[str, List[float]] = None,
                 **kwargs: dict) -> None:
        
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)
    
    def observe(self) -> np.ndarray:

        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        return obs.astype(self.space().dtype)

class DynamicGoalObservation(DynamicObservation):
    
    def __init__(self, 
                 env: "AbstractEnv",
                 **kwargs: dict) -> None:
        super().__init__(env, **kwargs)
        if hasattr(env, "goal"):
            self.goal = env.goal

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64)
            ))
        except AttributeError:
            return spaces.Space()
    
    def observe(self) -> Dict[str, np.ndarray]:
        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        # obs = obs.astype(self.space().dtype)
        obs = OrderedDict([
            ("observation", obs[0]),
            ("achieved_goal", obs[0][0:3]),
            ("desired_goal", self.goal)
        ])
        return obs


class LateralGoalObservation(DynamicObservation):

    FEATURES: List[str] = ['x', 'y',  'u', 'v', 'yaw']

    def __init__(self,
                 env: "AbstractEnv",
                 features: List[str] = None,
                 **kwargs: dict) -> None:
        super().__init__(env, **kwargs)
        self.features = features or self.FEATURES
        if hasattr(env, "goal"):
            self.goal = env.goal

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64)
            ))
        except AttributeError:
            return spaces.Space()
        
    def observe(self) -> Dict[str, np.ndarray]:
        df = pd.DataFrame.from_records([self.observer_vehicle.dict])[self.features]
        df = df[self.features]
        obs = df.values.copy()
        obs = OrderedDict([
            ("observation", obs[0]),
            ("achieved_goal", obs[0][0:2]),
            ("desired_goal", self.goal[0:2])
        ])
        return obs


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    if config["type"] == "Dynamics" or config["type"] == "dynamics":
        return DynamicObservation(env, **config)
    elif config["type"] == "Trajectory" or config["type"] == "trajectory":
        return TrajectoryObservation(env, **config)
    elif config["type"] == "LateralTrajectory" or config["lateral_trajectory"]:
        return LateralTrajectoryObservation(env, **config)
    elif config["type"] == "Control" or config["type"] == "control":
        return ControlObservation(env, **config)
    elif config["type"] == "Longitudinal" or config["type"] == "longitudinal":
        return LongitudinalObservation(env, **config)
    elif config["type"] == "Goal" or config["type"] == "goal" or config["type"] == "DynamicGoal":
        return DynamicGoalObservation(env, **config)
    elif config["type"] == "LateralGoal" or config["type"] == "lateral_goal":
        return LateralGoalObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
