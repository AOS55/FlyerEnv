import pytest
import numpy as np
from typing import Dict, Any

from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestFullAircraftEnvironment(BaseSingleAgentTest):
    """Test suite for full aircraft environment."""

    def get_config(self) -> Dict[str, Any]:
        """Override to use Full Aircraft config."""
        return EnvironmentConfigs.get_full_config()

    def test_reset(self):
        """Test basic reset functionality and initial state."""
        obs, info = self.env.reset()

        # Validate initial observation contains all required state variables
        expected_features = [
            "x", "y", "z",           # Position
            "roll", "pitch", "yaw",  # Attitude
            "u", "v", "w",           # Body-frame velocities
            "p", "q", "r"            # Angular rates
        ]
        assert self.verify_observation_features(obs, expected_features)

        # Test deterministic reset with seed
        seed = 42
        obs1, _ = self.env.reset(seed=seed)
        obs2, _ = self.env.reset(seed=seed)

        assert self.validate_state_consistency(obs1, obs2)

    def test_trim_conditions(self):
        """Test aircraft behavior near trim conditions."""
        obs, _ = self.env.reset()

        # Apply zero control inputs (should be near trim)
        action = np.zeros(4)  # [elevator, aileron, throttle, rudder]

        # Run for several steps to check stability
        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Check basic stability (angular rates should remain small)
            assert abs(obs['p']) < 0.1, "Roll rate too high in trim"
            assert abs(obs['q']) < 0.1, "Pitch rate too high in trim"
            assert abs(obs['r']) < 0.1, "Yaw rate too high in trim"

            assert not terminated, "Aircraft should maintain stable flight in trim"

    @pytest.mark.parametrize("control_input", [
        (1.0, 0, 0, 0),   # Pure elevator
        (0, 1.0, 0, 0),   # Pure aileron
        (0, 0, 1.0, 0),   # Pure throttle
        (0, 0, 0, 1.0),   # Pure rudder
    ])
    def test_control_surface_responses(self, control_input):
        """Test aircraft response to individual control inputs."""
        obs, _ = self.env.reset()
        initial_state = obs.copy()

        action = np.array(control_input)
        obs, _, _, _, _ = self.env.step(action)

        # Verify expected responses for each control input
        if control_input[0] != 0:  # Elevator
            assert obs['q'] != initial_state['q'], \
                "No pitch rate response to elevator"

        if control_input[1] != 0:  # Aileron
            assert obs['p'] != initial_state['p'], \
                "No roll rate response to aileron"

        if control_input[2] != 0:  # Throttle
            assert obs['u'] != initial_state['u'], \
                "No speed response to throttle"

        if control_input[3] != 0:  # Rudder
            assert obs['r'] != initial_state['r'], \
                "No yaw rate response to rudder"

    def test_coupled_dynamics(self):
        """Test coupled dynamic responses."""
        obs, _ = self.env.reset()

        # Apply combined control input
        action = np.array([0.5, 0.5, 0.5, 0])  # elevator + aileron + throttle
        obs, _, _, _, _ = self.env.step(action)

        # Check for expected coupled motions
        assert obs['p'] != 0, "No roll rate in coupled motion"
        assert obs['q'] != 0, "No pitch rate in coupled motion"
        assert obs['r'] != 0, "No yaw rate in coupled motion (from roll-yaw coupling)"

    def test_flight_envelope(self):
        """Test behavior near flight envelope limits."""
        self.env.reset()

        # Test high angle of attack
        high_alpha_action = np.array([1.0, 0, 0.5, 0])  # Pull up with moderate throttle

        # Run for several steps
        for _ in range(5):
            obs, _, terminated, _, _ = self.env.step(high_alpha_action)

            if terminated:
                # Check if termination was due to flight envelope exceedance
                assert obs['q'] > 1.0 or obs['w']/obs['u'] > 0.5, \
                    "Unexpected termination in flight envelope test"
                break

    @pytest.mark.parametrize("maneuver", [
        ("level_turn", np.array([0.2, 0.3, 0.7, 0.1])),   # Coordinated turn
        ("climb", np.array([0.3, 0, 0.8, 0])),            # Steady climb
        ("descent", np.array([-0.2, 0, 0.3, 0])),         # Steady descent
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
            assert obs['yaw'] != initial_state['yaw'], \
                "No heading change in turn"
            assert abs(obs['z'] - initial_state['z']) < 10.0, \
                "Significant altitude change in level turn"

        elif name == "climb":
            assert obs['z'] < initial_state['z'], \
                "No altitude gain in climb"  # Remember NED coordinates

        elif name == "descent":
            assert obs['z'] > initial_state['z'], \
                "No altitude loss in descent"  # Remember NED coordinates

    def test_energy_conservation(self):
        """Test basic energy relationships."""
        obs, _ = self.env.reset()

        # Calculate initial energy state
        initial_ke = 0.5 * (obs['u']**2 + obs['v']**2 + obs['w']**2)  # Kinetic energy
        initial_pe = -9.81 * obs['z']  # Potential energy (NED frame)

        # Apply zero thrust and check energy decay
        action = np.array([0, 0, 0, 0])  # No thrust
        obs, _, _, _, _ = self.env.step(action)

        # Calculate new energy state
        new_ke = 0.5 * (obs['u']**2 + obs['v']**2 + obs['w']**2)
        new_pe = -9.81 * obs['z']

        # Total energy should decrease due to drag
        assert (new_ke + new_pe) < (initial_ke + initial_pe), \
            "Energy increased without thrust input"

    def test_state_consistency(self):
        """Test physical consistency of state variables"""
        obs, _ = self.env.reset()

        for _ in range(10):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)

            if not terminated:
                # Check basic physical constraints
                velocity_magnitude = np.sqrt(obs['u']**2 + obs['v']**2 + obs['w']**2)
                assert velocity_magnitude > 0, "Zero velocity in flight"

                # Check euler angles are within reasonable range
                assert -np.pi <= obs['roll'] <= np.pi, "Roll angle out of range"
                assert -np.pi/2 <= obs['pitch'] <= np.pi/2, "Pitch angle out of range"
                assert -np.pi <= obs['yaw'] <= np.pi, "Yaw angle out of range"
