import numpy as np

class ObservationValidator:
    @staticmethod
    def validate(obs: np.ndarray, low: np.ndarray = None, high: np.ndarray = None) -> bool:
        """Validate observation array structure, values, and optional bounds."""
        assert isinstance(obs, np.ndarray), "Observation must be a NumPy array."
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values (e.g., inf or NaN)."

        if low is not None and high is not None:
            assert obs.shape == low.shape == high.shape, "Observation and bounds must have the same shape."
            assert np.all(obs >= low), f"Observation values are below the lower bounds: {low}."
            assert np.all(obs <= high), f"Observation values exceed the upper bounds: {high}."

        return True

class ActionValidator:
    @staticmethod
    def validate(action: np.ndarray, action_space_low: np.ndarray,
                action_space_high: np.ndarray) -> bool:
        """Validate action array against action space bounds."""
        assert np.all(action >= action_space_low), "Action below lower bound"
        assert np.all(action <= action_space_high), "Action above upper bound"
        assert np.all(np.isfinite(action)), "Non-finite values in action"
        return True

class StateValidator:
    @staticmethod
    def check_consistency(state1: np.ndarray, state2: np.ndarray,
                         rtol=1e-5, atol=1e-8) -> bool:
        """Check if two states are consistent within tolerance."""
        return np.allclose(state1, state2, rtol=rtol, atol=atol)
