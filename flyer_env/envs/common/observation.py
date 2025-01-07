from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from gymnasium import spaces

class ObservationType:
    """Base class for all observation types following Gymnasium conventions"""

    @property
    def space(self) -> spaces.Space:
        """The observation space"""
        raise NotImplementedError

    def observe(self):
        """Process raw observation into proper format"""
        raise NotImplementedError

class DubinsObservation(ObservationType):
    """Observation space for Dubins aircraft with optional normalization"""

    def __init__(self,
        normalize: bool = False,
        obs_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None
    ) -> None:
        """
        Initialize Dubins aircraft observation space.

        Args:
            normalize: If True, normalize observations to [-1, 1]
            obs_bounds: Optional bounds for each observation dimension.
                If None for a dimension, that dimension is unbounded.
                Format: {"obs_name": (min_val, max_val) or None}
        """

        self.normalize = normalize
        self.features = ["x", "y", "heading", "altitude", "airspeed"]

        self._default_bounds = {
            "x": None,  # Unbounded
            "y": None,  # Unbounded
            "heading": (-np.pi, np.pi),
            "altitude": (0, 20000),  # 0 to 20 km
            "airspeed": (0, 900)  # 0 to 900 m/s
        }

        self.obs_bounds = self._default_bounds.copy()
        if obs_bounds:
            self.obs_bounds.update(obs_bounds)

    @property
    def space(self) -> spaces.Space:
        """The observation space for the Dubins aircraft"""
        if self.normalize:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.features),),
                dtype=np.float32
            )
        else:
            lows = []
            highs = []
            for feature in self.features:
                bounds = self.obs_bounds[feature]
                if bounds is None:
                    lows.append(-np.inf)
                    highs.append(np.inf)
                else:
                    lows.append(bounds[0])
                    highs.append(bounds[1])

            return spaces.Box(
                low=np.array(lows),
                high=np.array(highs),
                dtype=np.float32
            )


    def observe(self, raw_obs: Dict[str, float]) -> np.ndarray:
        """
        Convert raw observation dict into normalized or unnormalized numpy array.

        Args:
            raw_obs: Dictionary of raw observation values

        Returns:
            Processed observation array
        """
        obs = []
        for feature in self.features:
            value = raw_obs.get(feature, 0.0)  # Default to 0.0 if missing
            print(f"Feature {feature}: {value} (raw: {raw_obs.get(feature)})")

            if self.normalize:
                bounds = self.obs_bounds[feature]
                if bounds is None:
                    # Keep unbounded values as is
                    obs.append(value)
                else:
                    # Normalize bounded values to [-1, 1]
                    low, high = bounds
                    if high > low:  # Avoid division by zero
                        value = 2.0 * (value - low) / (high - low) - 1.0
                    obs.append(value)
            else:
                obs.append(value)

        return np.array(obs, dtype=np.float32)

class FullObservation(ObservationType):
    """Observation space for full aircraft state"""

    def __init__(
        self,
        normalize: bool = True,
        obs_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None
    ):
        """
        Initialize full aircraft observation space.

        Args:
            normalize: If True, normalize observations to [-1, 1]
            obs_bounds: Optional bounds for each observation dimension
        """
        self.normalize = normalize
        self.features = [
            "x", "y", "z",           # Position
            "roll", "pitch", "yaw",  # Orientation
            "u", "v", "w",           # Linear velocities
            "p", "q", "r"            # Angular velocities
        ]

        self._default_bounds = {
            # Position bounds
            "x": None,
            "y": None,
            "z": None,
            # Orientation bounds
            "roll": (-np.pi, np.pi),
            "pitch": (-0.5 * np.pi, 0.5 * np.pi),
            "yaw": (-np.pi, np.pi),
            # Linear velocity bounds
            "u": (-100, 100),
            "v": (-50, 50),
            "w": (-50, 50),
            # Angular velocity bounds
            "p": (-2*np.pi, 2*np.pi),
            "q": (-2*np.pi, 2*np.pi),
            "r": (-2*np.pi, 2*np.pi)
        }

        self.obs_bounds = self._default_bounds.copy()
        if obs_bounds:
            self.obs_bounds.update(obs_bounds)

    @property
    def space(self) -> spaces.Space:
        """The observation space following Gymnasium conventions"""
        if self.normalize:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.features),),
                dtype=np.float32
            )
        else:
            lows = []
            highs = []
            for feature in self.features:
                bounds = self.obs_bounds[feature]
                if bounds is None:
                    lows.append(-np.inf)
                    highs.append(np.inf)
                else:
                    lows.append(bounds[0])
                    highs.append(bounds[1])

            return spaces.Box(
                low=np.array(lows),
                high=np.array(highs),
                dtype=np.float32
            )

    def observe(self, raw_obs: Dict[str, float]) -> np.ndarray:
        """
        Convert raw observation dict into normalized or unnormalized numpy array.

        Args:
            raw_obs: Dictionary of raw observation values

        Returns:
            Processed observation array
        """
        obs = []
        for feature in self.features:
            value = raw_obs.get(feature, 0.0)  # Default to 0.0 if missing

            if self.normalize:
                bounds = self.obs_bounds[feature]
                if bounds is None:
                    # Keep unbounded values as is
                    obs.append(value)
                else:
                    # Normalize bounded values to [-1, 1]
                    low, high = bounds
                    if high > low:  # Avoid division by zero
                        value = 2.0 * (value - low) / (high - low) - 1.0
                    obs.append(value)
            else:
                obs.append(value)

        return np.array(obs, dtype=np.float32)

def observation_factory(aircraft_type: str, observation_type: str, **kwargs) -> ObservationType:
    """
    Create an observation space based on aircraft and observation type.

    Args:
        aircraft_type: Type of aircraft ("Dubins" or "Full")
        observation_type: Type of observation space ("Continuous" currently)
        **kwargs: Additional arguments passed to observation space constructor

    Returns:
        Appropriate ObservationType instance

    Raises:
        ValueError: If aircraft_type or observation_type is invalid
    """
    if aircraft_type == "Dubins":
        if observation_type == "Continuous":
            return DubinsObservation(**kwargs)
        else:
            raise ValueError(f"Invalid observation type for Dubins aircraft: {observation_type}")

    elif aircraft_type == "Full":
        if observation_type == "Continuous":
            return FullObservation(**kwargs)
        else:
            raise ValueError(f"Invalid observation type for Full aircraft: {observation_type}")

    else:
        raise ValueError(f"Invalid aircraft type: {aircraft_type}")
