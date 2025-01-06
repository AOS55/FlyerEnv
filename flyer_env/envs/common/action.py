import functools
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union, List

import numpy as np
from gymnasium import spaces

Action = Union[int, np.ndarray]

class ActionType:

    def __init__(self) -> None:
        return

    def space(self) -> spaces.Space:
        """The action space"""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        "Format the action to be used in the enviornment and send it out to be processed"
        raise NotImplementedError

class DubinsContinuousAction(ActionType):

    FEATURES: List[str] = [
        "acceleration",
        "bank_angle",
        "vertical_speed"
    ]

    def __init__(self, features_range: Dict[str, List[float]] = None, normalize: bool = False) -> None:
        """Initialize the Dubins aircraft action type"""
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
        """Action space for the Dubins aircraft"""
        if self.normalize:
            return spaces.Box(low=-1, high=1, shape=(len(self.features),))
        else:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),))

    def act(self, action: Action) -> None:

        if len(action) != len(self.features):
            raise ValueError(f"Action vector has {len(action)} elements, expected {len(self.features)}")

        # Format the action dictionary
        action_values = [float(a) for a in action]  # TODO: This isn't ideal review why this is needed?

        # Normalize the action if needed
        if self.normalize:
            action_values = [(val + 1) / 2 * (max_val - min_val) + min_val
            for val, (min_val, max_val) in zip(action_values, [self.features_range[f] for f in self.features])]

        # Send the action to the environment
        return action_values

class DubinsDiscreteAction(ActionType):
    # TODO: Implement the DubinsDiscreteAction class

    FEATURES: List[str] = [
        "acceleration",
        "bank_angle",
        "vertical_speed"
    ]

    def __init__(self, features_range: Dict[str, List[float]] = None) -> None:
        """Initialize the Dubins aircraft action type"""
        if features_range:
            self.features = list(features_range.keys())
            self.features_range = features_range
        else:
            self.features = self.FEATURES

    def space(self) -> spaces.Space:
        """Action space for the Dubins aircraft"""
        return spaces.MultiDiscrete([len(self.features_range[feature]) for feature in self.features])

    def act(self, action: Action) -> Dict[str, float]:
        raise NotImplementedError

class FullContinuousAction(ActionType):
    # TODO: Implement the FullContinuousAction class

    FEATURES: List[str] = [
        "aileron",
        "elevator",
        "throttle",
        "rudder"
    ]

    def __init__(self):
        return

    def space(self) -> spaces.Space:
        return spaces.Box(low=-1, high=1, shape=len(self.FEATURES))

    def act(self, action: Action) -> Dict[str, float]:
        raise NotImplementedError

class FullDiscreteAction(ActionType):
    # TODO: Implement the FullDiscreteAction class
    FEATURES: List[str] = [
        "aileron",
        "elevator",
        "throttle",
        "rudder"
    ]

    def __init__(self):
        return

    def space(self) -> spaces.Space:
        return spaces.MultiDiscrete([3, 3, 3, 3])

    def acti(self, action: Action) -> Dict[str, float]:
        raise NotImplementedError


def action_factory(aircraft_type: str, action_type: str, **kwargs) -> ActionType:
    if aircraft_type == "Dubins":
        if action_type == "Continuous":
            return DubinsContinuousAction(**kwargs)
        elif action_type == "Discrete":
            return DubinsDiscreteAction(**kwargs)
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    elif aircraft_type == "Full":
        if action_type == "Continuous":
            return FullContinuousAction(**kwargs)
        elif action_type == "Discrete":
            return FullDiscreteAction(**kwargs)
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    else:
        raise ValueError(f"Invalid aircraft type: {aircraft_type}")
