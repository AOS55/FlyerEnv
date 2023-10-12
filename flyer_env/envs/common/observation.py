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


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    if config["type"] == "Dynamics" or config["type"] == "dynamics":
        return DynamicObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
