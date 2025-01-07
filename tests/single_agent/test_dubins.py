import pytest
import numpy as np
from typing import Dict, Any

from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestDubinsEnvironment(BaseSingleAgentTest):
    """Test suite for Dubins aircraft environment."""

    def get_config(self) -> Dict[str, Any]:
        """Override to use Dubins config."""
        return EnvironmentConfigs.get_dubins_config()

    def test_reset(self):
        """Test basic reset functionality and initial state."""
        obs, info = self.env.reset()

        # Validate initial observation contains all required state variables
        expected_features = [
            "x", "y",              # Position
            "heading",             # Orientation
            "altitude",            # Altitude
            "airspeed"            # Velocity
        ]
        assert self.verify_observation_features(obs, expected_features)

        # Test deterministic reset with seed
        seed = 42
        obs1, _ = self.env.reset(seed=seed)
        obs2, _ = self.env.reset(seed=seed)

        assert self.validate_state_consistency(obs1, obs2)

    @pytest.mark.parametrize("control_input", [
        (1.0, 0, 0),    # Pure acceleration
        (0, 1.0, 0),    # Pure bank angle
        (0, 0, 1.0),    # Pure vertical speed
    ])
    def test_control_responses(self, control_input):
        """Test aircraft response to individual control inputs."""
        obs, _ = self.env.reset()
        initial_state = obs.copy()

        action = np.array(control_input)
        obs, _, terminated, _, _ = self.env.step(action)

        # Verify expected responses for each control input
        if control_input[0] != 0:  # Acceleration
            assert obs['airspeed'] != initial_state['airspeed'], \
                "No airspeed response to acceleration"

        if control_input[1] != 0:  # Bank angle
            assert obs['heading'] != initial_state['heading'], \
                "No heading response to bank angle"

        if control_input[2] != 0:  # Vertical speed
            assert obs['altitude'] != initial_state['altitude'], \
                "No altitude response to vertical speed command"

    def test_flight_envelope(self):
        """Test behavior near flight envelope limits."""
        self.env.reset()

        # Test high speed
        high_speed_action = np.array([1.0, 0, 0])  # Maximum acceleration

        # Run for several steps
        for _ in range(5):
            obs, _, terminated, _, _ = self.env.step(high_speed_action)

            if terminated:
                assert obs['airspeed'] > self.env.config['max_airspeed'], \
                    "Unexpected termination in flight envelope test"
                break

    @pytest.mark.parametrize("maneuver", [
        ("level_turn", np.array([0.5, 0.5, 0])),    # Turning flight
        ("climb", np.array([0.5, 0, 0.5])),         # Climbing flight
        ("descent", np.array([0.5, 0, -0.5])),      # Descending flight
    ])
    def test_basic_maneuvers(self, maneuver):
        """Test basic flight maneuvers."""
        name, action = maneuver
        obs, _ = self.env.reset()
        initial_state = obs.copy()

        # Execute maneuver for several steps
        steps = 10
        for _ in range(steps):
            obs, _, terminated, _, _ = self.env.step(action)
            assert not terminated, f"Aircraft terminated during {name} maneuver"

        # Verify expected state changes for each maneuver
        if name == "level_turn":
            assert obs['heading'] != initial_state['heading'], \
                "No heading change in turn"
            assert abs(obs['altitude'] - initial_state['altitude']) < 10.0, \
                "Significant altitude change in level turn"

        elif name == "climb":
            assert obs['altitude'] > initial_state['altitude'], \
                "No altitude gain in climb"

        elif name == "descent":
            assert obs['altitude'] < initial_state['altitude'], \
                "No altitude loss in descent"

    def test_state_consistency(self):
        """Test physical consistency of state variables."""
        obs, _ = self.env.reset()

        for _ in range(10):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)

            if not terminated:
                # Check altitude constraints
                assert -10000 <= obs['altitude'] <= 0, \
                    "Altitude out of reasonable range"

                # Check airspeed constraints
                assert 0 <= obs['airspeed'] <= 200, \
                    "Airspeed out of reasonable range"

                # Check heading normalization
                assert -np.pi <= obs['heading'] <= np.pi, \
                    "Heading angle not normalized"

    def test_steady_flight(self):
        """Test steady flight conditions."""
        obs, _ = self.env.reset()

        # Apply neutral controls
        steady_action = np.zeros(3)

        # Run for several steps
        n_steps = 20
        altitude_history = []
        heading_history = []
        speed_history = []

        for _ in range(n_steps):
            obs, _, _, _, _ = self.env.step(steady_action)
            altitude_history.append(obs['altitude'])
            heading_history.append(obs['heading'])
            speed_history.append(obs['airspeed'])

        # Check stability of flight parameters
        assert max(altitude_history) - min(altitude_history) < 5.0, \
            "Excessive altitude variation in steady flight"
        assert max(speed_history) - min(speed_history) < 2.0, \
            "Excessive speed variation in steady flight"
        assert max(heading_history) - min(heading_history) < 0.1, \
            "Excessive heading variation in steady flight"

    def test_boundary_conditions(self):
        """Test environment boundaries and constraints."""
        obs, _ = self.env.reset()

        # Test minimum altitude limit
        min_alt_action = np.array([0.5, 0, -1.0])  # Descend at maximum rate
        while obs['altitude'] > -100:  # Arbitrary low altitude
            obs, _, terminated, _, _ = self.env.step(min_alt_action)
            if terminated:
                break
        assert terminated, "No termination at minimum altitude"

        # Test maximum speed limit
        self.env.reset()
        max_speed_action = np.array([1.0, 0, 0])  # Maximum acceleration
        while obs['airspeed'] < 150:  # Arbitrary high speed
            obs, _, terminated, _, _ = self.env.step(max_speed_action)
            if terminated:
                break
        assert terminated, "No termination at maximum speed"
