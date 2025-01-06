import numpy as np
from gymnasium import spaces

from typing import List, Dict

class ObservationType:

    def __init__(self) -> None:
        return

    def space(self) -> spaces.Space:
        """The observation space"""
        raise NotImplementedError

    def observe(self):
        "Get an observation of the environment"
        raise NotImplementedError

class DubinsObservation(ObservationType):

    FEATURES: List[str] = [
        "x",
        "y",
        "heading",
        "altitude",
        "airspeed"
    ]

    def __init__(self, features_range: Dict[str, List[float]] = None, normalize: bool = False) -> None:
        """Initialize the Dubins aircraft observation type"""
        if (features_range and normalize):
            self.normalize = True
        else:
            self.normalize = False

        if features_range:
            self.features = list(features_range.keys())
            self.features_range = features_range
        else:
            self.features = self.FEATURES

    def space(self) -> spaces.Space:
        """Observation space for the Dubins aircraft"""
        if self.normalize:
            return spaces.Box(low=-1, high=1, shape=len(self.features))
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=len(self.features))

    def observe(self, obs_dict: Dict[str, float]) -> np.ndarray:
        """Generate observation vector from the observation dictionary."""
        obs_vector = []
        for feature in self.features:
            value = obs_dict.get(feature, 0.0)  # Default to 0.0 if the feature is missing
            if self.normalize:
                # Perform normalization using the range
                feature_range = self.features_range.get(feature, [-np.inf, np.inf])
                min_val, max_val = feature_range
                if max_val > min_val:  # Avoid division by zero
                    value = 2 * (value - min_val) / (max_val - min_val) - 1
                else:
                    value = 0.0  # Handle edge cases where min and max are the same
            obs_vector.append(value)
        return np.array(obs_vector)

class FullObservation(ObservationType):

    FEATURES: List[str] = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
    ]

    def __init__(self, features_range: Dict[str, List[float]] = None, normalize: bool = False) -> None:
        """Initialize the Full aircraft observation type"""
        if (features_range and normalize):
            self.normalize = True
        else:
            self.normalize = False

        if features_range:
            self.features = list(features_range.keys())
            self.features_range = features_range
        else:
            self.features = self.FEATURES

    def space(self) -> spaces.Space:
        """Observation space for the Dubins aircraft"""
        if self.normalize:
            return spaces.Box(low=-1, high=1, shape=len(self.features))
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=len(self.features))

    def observe(self, obs_dict):
        """Generate observation vector from the observation dictionary."""
        obs_vector = []
        for feature in self.features:
            value = obs_dict.get(feature, 0.0)  # Default to 0.0 if the feature is missing
            if self.normalize:
                # Perform normalization using the range
                feature_range = self.features_range.get(feature, [-np.inf, np.inf])
                min_val, max_val = feature_range
                if max_val > min_val:  # Avoid division by zero
                    value = 2 * (value - min_val) / (max_val - min_val) - 1
                else:
                    value = 0.0  # Handle edge cases where min and max are the same
            obs_vector.append(value)
        return np.array(obs_vector)

def observation_factory(aircraft_type: str, observation_type: str, **kwargs) -> ObservationType:
    if aircraft_type == "Dubins":
        if observation_type == "Continuous":
            return DubinsObservation(**kwargs)
        else:
            raise ValueError(f"Invalid observation type: {observation_type}")

    elif aircraft_type == "Full":
        if observation_type == "Continuous":
            return FullObservation(**kwargs)
        else:
            raise ValueError(f"Invalid observation type: {observation_type}")

    else:
        raise ValueError(f"Invalid aircraft type: {aircraft_type}")
