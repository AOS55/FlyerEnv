import numpy as np

class ObservationValidator:
    @staticmethod
    def validate(obs: dict, expected_features: list) -> bool:
        """Validate observation dictionary structure and values."""
        for feature in expected_features:
            assert feature in obs, f"Missing feature: {feature}"
        for value in obs.values():
            assert np.isfinite(value), f"Non-finite value in observation: {value}"
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
    def check_consistency(state1: dict, state2: dict, 
                         rtol=1e-5, atol=1e-8) -> bool:
        """Check if two states are consistent within tolerance."""
        assert state1.keys() == state2.keys(), "States have different keys"
        for key in state1:
            np.testing.assert_allclose(
                state1[key], state2[key],
                rtol=rtol, atol=atol,
                err_msg=f"Inconsistent values for key: {key}"
            )
        return True