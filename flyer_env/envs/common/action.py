from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from gymnasium import spaces

Action = Union[int, np.ndarray]


class ActionType:
    """Base class for all action types"""

    @property
    def space(self) -> spaces.Space:
        """The action space following gymnasium conventions"""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """Process action and return formatted dictionary for server"""
        raise NotImplementedError


class DubinsContinuousAction(ActionType):
    """Continuous action space for Dubins aircraft"""

    def __init__(self,
        normalize: bool = True,
        action_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None
    ) -> None:
        """
        Initialize Dubins aircraft continuous action space.

        Args:
            normalize: If true, normalize actions to [-1, +1]
            action_bounds: Optional bounds for each action dimension.
                If None for a dimension, the dimension is unbounded.

        """

        self.normalize = normalize
        self.features = ["acceleration", "bank_angle", "vertical_speed"]

        self._default_bounds = {
            "acceleration": None,
            "bank_angle": (-np.pi, np.pi),
            "vertical_speed": None,
        }

        self.action_bounds = self._default_bounds.copy()
        if action_bounds:
            self.action_bounds.update(action_bounds)

    @property
    def space(self) -> spaces.Space:
        """Action space for the Dubins aircraft"""
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
                bounds = self.action_bounds[feature]
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

    def from_dict(self, action_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert dictionary of unnormalized actions to numpy array.

        Args:
            action_dict: Dictionary mapping feature names to their raw values

        Returns:
            Numpy array of actions (normalized if self.normalize is True)
        """
        action = np.zeros(len(self.features), dtype=np.float32)

        for i, feature in enumerate(self.features):
            if feature not in action_dict:
                raise ValueError(f"Missing required feature: {feature}")

            value = float(action_dict[feature])
            bounds = self.action_bounds[feature]

            if self.normalize:
                if bounds is None:
                    # For unbounded actions, use value directly
                    action[i] = value
                else:
                    # Normalize to [-1, 1]
                    min_val, max_val = bounds
                    action[i] = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
            else:
                if bounds is not None:
                    # Clip to bounds if not normalizing
                    min_val, max_val = bounds
                    value = np.clip(value, min_val, max_val)
                action[i] = value

        return action

    def to_dict(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert action array back to dictionary with original values.

        Args:
            action: Numpy array of actions (normalized or unnormalized)

        Returns:
            Dictionary mapping feature names to their original values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        result = {}
        for i, feature in enumerate(self.features):
            value = float(action[i])

            if self.normalize:
                bounds = self.action_bounds[feature]
                if bounds is None:
                    # For unbounded actions, use the normalized value directly
                    result[feature] = value
                else:
                    # Denormalize bounded values from [-1, 1]
                    min_val, max_val = bounds
                    result[feature] = min_val + (value + 1.0) * 0.5 * (max_val - min_val)
            else:
                bounds = self.action_bounds[feature]
                if bounds is None:
                    result[feature] = value
                else:
                    # Clip to bounds if not normalized
                    min_val, max_val = bounds
                    result[feature] = float(np.clip(value, min_val, max_val))

        return result

    def act(self, action: np.ndarray) -> Dict[str, float]:
        """
        Process action array into a dictionary of values.

        Args:
            action: Array of action values, either normalized [-1, 1] or raw values.

        Returns:
            Dictionary mapping feature names to processed values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        processed_dict = {}

        if self.normalize:
            # Denormalize actions
            for ida, feature in enumerate(self.features):
                bounds = self.action_bounds[feature]
                if bounds is None:
                    # For unbounded actions, use the normalized value directly
                    processed_dict[feature] = float(action[ida])
                else:
                    min_val, max_val = bounds
                    processed_dict[feature] = float(
                        min_val + (action[ida] + 1.0) * 0.5 * (max_val - min_val)
                    )
        else:
            # When not normalizing, use raw values but respect bounds
            for ida, feature in enumerate(self.features):
                bounds = self.action_bounds[feature]
                if bounds is None:
                    processed_dict[feature] = float(action[ida])
                else:
                    min_val, max_val = bounds
                    processed_dict[feature] = float(np.clip(action[ida], min_val, max_val))

        return processed_dict


class DubinsDiscreteAction(ActionType):
    """Discrete action space for Dubins aircraft"""

    def __init__(self, action_values: Optional[Dict[str, List[float]]] = None):
        """
        Initialize discrete action space for Dubins aircraft.

        Args:
            action_values: Optional discrete values for each action dimension.
                Format: {"action_name": [value1, value2, ...]}
        """
        self.features = ["acceleration", "bank_angle", "vertical_speed"]

        # Artficially constrain if no value provided
        self._default_values = {
            "acceleration": [-1.0, 0.0, 1.0],
            "bank_angle": [-np.pi, 0.0, np.pi],
            "vertical_speed": [-5.0, 0.0, 5.0]
        }

        self.action_values = self._default_values.copy()
        if action_values:
            self.action_values.update(action_values)

    @property
    def space(self) -> spaces.Space:
        """The discrete action space"""
        return spaces.MultiDiscrete([
            len(self.action_values[feature])
            for feature in self.features
        ])

    def from_dict(self, action_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert dictionary of continuous values to discrete action indices.

        Args:
            action_dict: Dictionary mapping feature names to their continuous values

        Returns:
            Numpy array of discrete action indices
        """
        action = np.zeros(len(self.features), dtype=np.int64)

        for i, feature in enumerate(self.features):
            if feature not in action_dict:
                raise ValueError(f"Missing required feature: {feature}")

            value = float(action_dict[feature])
            values = np.array(self.action_values[feature])

            # Find closest discrete value
            idx = np.abs(values - value).argmin()
            action[i] = idx

        return action

    def to_dict(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert discrete action indices to dictionary with actual values.

        Args:
            action: Array of discrete action indices

        Returns:
            Dictionary mapping feature names to their actual values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int64)

        result = {}
        for i, feature in enumerate(self.features):
            values = self.action_values[feature]
            idx = int(action[i])
            if not 0 <= idx < len(values):
                raise ValueError(
                    f"Invalid action index {idx} for feature {feature}. "
                    f"Must be between 0 and {len(values)-1}"
                )
            result[feature] = float(values[idx])

        return result

    def act(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert discrete actions to continuous values.

        Args:
            action: Array of discrete action indices

        Returns:
            Dictionary mapping feature names to continuous values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int64)

        processed_dict = {}
        for i, feature in enumerate(self.features):
            values = self.action_values[feature]
            idx = int(action[i])
            if not 0 <= idx < len(values):
                raise ValueError(
                    f"Invalid action index {idx} for feature {feature}. "
                    f"Must be between 0 and {len(values)-1}"
                )
            processed_dict[feature] = float(values[idx])
        return processed_dict


class FullContinuousAction(ActionType):
    """Continuous action space for full aircraft model"""

    def __init__(
        self,
        normalize: bool = True,
        action_bounds: Optional[Dict[str, Optional[Tuple[float, float]]]] = None
    ):
        self.normalize = normalize
        self.features = ["elevator", "aileron", "throttle", "rudder"]

        self._default_bounds = {
            "elevator": (-0.5 * np.pi, 0.5 * np.pi),
            "aileron": (-0.5 * np.pi, 0.5 * np.pi),
            "throttle": (0.0, 1.0),
            "rudder": (-0.5 * np.pi, 0.5 * np.pi)
        }

        self.action_bounds = self._default_bounds.copy()
        if action_bounds:
            self.action_bounds.update(action_bounds)

    @property
    def space(self) -> spaces.Space:
        """The action space following Gymnasium conventions"""
        if self.normalize:
            return spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.features),),
                dtype=np.float32
            )
        else:
            lows = [b[0] for b in self.action_bounds.values()]
            highs = [b[1] for b in self.action_bounds.values()]
            return spaces.Box(
                low=np.array(lows),
                high=np.array(highs),
                dtype=np.float32
            )

    def from_dict(self, action_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert dictionary of unnormalized actions to numpy array.

        Args:
            action_dict: Dictionary mapping feature names to their raw values

        Returns:
            Numpy array of actions (normalized if self.normalize is True)
        """
        action = np.zeros(len(self.features), dtype=np.float32)

        for i, feature in enumerate(self.features):
            if feature not in action_dict:
                raise ValueError(f"Missing required feature: {feature}")

            value = float(action_dict[feature])
            bounds = self.action_bounds[feature]

            if self.normalize:
                # Normalize to [-1, 1]
                min_val, max_val = bounds
                action[i] = 2.0 * (value - min_val) / (max_val - min_val) - 1.0
            else:
                # Clip to bounds
                min_val, max_val = bounds
                action[i] = np.clip(value, min_val, max_val)

        return action

    def to_dict(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert action array back to dictionary with original values.

        Args:
            action: Numpy array of actions (normalized or unnormalized)

        Returns:
            Dictionary mapping feature names to their original values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        result = {}
        for i, feature in enumerate(self.features):
            value = float(action[i])
            bounds = self.action_bounds[feature]

            if self.normalize:
                # Denormalize from [-1, 1] to actual range
                low, high = bounds
                result[feature] = low + (value + 1.0) * 0.5 * (high - low)
            else:
                # Clip to bounds if not normalized
                low, high = bounds
                result[feature] = float(np.clip(value, low, high))

        return result

    def act(self, action: np.ndarray) -> Dict[str, float]:
        """Process continuous actions for full aircraft model"""
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        processed_dict = {}
        if self.normalize:
            for ida, feature in enumerate(self.features):
                bounds = self.action_bounds[feature]
                low, high = bounds
                processed_dict[feature] = float(
                    low + (action[ida] + 1.0) * 0.5 * (high - low)
                )
        else:
            for ida, feature in enumerate(self.features):
                bounds = self.action_bounds[feature]
                processed_dict[feature] = float(
                    np.clip(action[ida], bounds[0], bounds[1])
                )
        return processed_dict


class FullDiscreteAction(ActionType):
    """Discrete action space for full aircraft model"""

    def __init__(self, action_values: Optional[Dict[str, List[float]]] = None):
        self.features = ["elevator", "aileron", "throttle", "rudder"]

        self._default_values = {
            "elevator": [-0.5 * np.pi, 0.0, 0.5 * np.pi],
            "aileron": [-0.5 * np.pi, 0.0, 0.5 * np.pi],
            "throttle": [0.0, 0.5, 1.0],
            "rudder": [-0.5 * np.pi, 0.0, 0.5 * np.pi]
        }

        self.action_values = self._default_values.copy()
        if action_values:
            self.action_values.update(action_values)

    @property
    def space(self) -> spaces.Space:
        """The discrete action space"""
        return spaces.MultiDiscrete([
            len(self.action_values[feature])
            for feature in self.features
        ])

    def from_dict(self, action_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert dictionary of continuous values to discrete action indices.

        Args:
            action_dict: Dictionary mapping feature names to their continuous values

        Returns:
            Numpy array of discrete action indices
        """
        action = np.zeros(len(self.features), dtype=np.int64)

        for i, feature in enumerate(self.features):
            if feature not in action_dict:
                raise ValueError(f"Missing required feature: {feature}")

            value = float(action_dict[feature])
            values = np.array(self.action_values[feature])

            # Find closest discrete value
            idx = np.abs(values - value).argmin()
            action[i] = idx

        return action

    def to_dict(self, action: np.ndarray) -> Dict[str, float]:
        """
        Convert discrete action indices to dictionary with actual values.

        Args:
            action: Array of discrete action indices

        Returns:
            Dictionary mapping feature names to their actual values
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int64)

        result = {}
        for i, feature in enumerate(self.features):
            values = self.action_values[feature]
            idx = int(action[i])
            if not 0 <= idx < len(values):
                raise ValueError(
                    f"Invalid action index {idx} for feature {feature}. "
                    f"Must be between 0 and {len(values)-1}"
                )
            result[feature] = float(values[idx])

        return result

    def act(self, action: np.ndarray) -> Dict[str, float]:
        """Convert discrete actions to continuous values"""
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.int64)

        processed_dict = {}
        for ida, feature in enumerate(self.features):
            values = self.action_values[feature]
            idx = int(action[ida])
            if not 0 <= idx < len(values):
                raise ValueError(
                    f"Invalid action index {idx} for feature {feature}. "
                    f"Must be between 0 and {len(values)-1}"
                )
            processed_dict[feature] = float(values[idx])
        return processed_dict


def action_factory(aircraft_type: str, action_type: str, config: Optional[Dict] = None, **kwargs) -> ActionType:

    if aircraft_type == "Dubins":

        if config:
            # Extract bounds from Dubins config
            action_bounds = {
                "acceleration": (-config["acceleration"], config["acceleration"]),
                "bank_angle": (-config["max_bank_angle"], config["max_bank_angle"]),
                "vertical_speed": (-config["max_descent_rate"], config["max_climb_rate"])
            }
            kwargs["action_bounds"] = action_bounds

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
