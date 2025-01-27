import pytest
import numpy as np
from typing import Dict, Any

from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestTaskTermination(BaseSingleAgentTest):
    """Test suite for different task termination conditions."""

    def get_config(self) -> Dict[str, Any]:
        """Get base config with default task configuration."""
        config = EnvironmentConfigs.get_dubins_config()

        # Add default task configuration
        config['aircraft_config'][0]['task_config'] = {
            "type": "Control",
            "config": {
                "control_type": "Heading",
                "target": 0.0,
                "tolerance": 0.1
            }
        }

        return config

    def setup_task_config(self, task_type: str, **kwargs):
        """Helper to configure task for testing."""
        config = self.get_config()

        # Configure task based on type and parameters
        if task_type == "Control":
            config['aircraft_config'][0]['task_config'] = {
                "type": "Control",
                "config": {
                    "control_type": kwargs.get("control_type", "Heading"),
                    "target": kwargs.get("target", 0.0),
                    "tolerance": kwargs.get("tolerance", 0.1)
                }
            }
        elif task_type == "Goal":
            config['aircraft_config'][0]['task_config'] = {
                "type": "Goal",
                "config": {
                    "position": kwargs.get("position", {"x": 1000.0, "y": 1000.0, "z": -500.0}),
                    "tolerance": kwargs.get("tolerance", 100.0)
                }
            }
        elif task_type == "Trajectory":
            config['aircraft_config'][0]['task_config'] = {
                "type": "Trajectory",
                "config": {
                    "target_velocity": kwargs.get("target_velocity", 20.0),
                    "motion_type": kwargs.get("motion_type", {
                        "type": "StraightAndLevel",
                        "target_distance": 100.0
                    })
                }
            }
        elif task_type == "Runway":
            config['aircraft_config'][0]['task_config'] = {
                "type": "Runway",
                "config": {
                    "position": kwargs.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                    "heading": kwargs.get("heading", 0.0),
                    "width": kwargs.get("width", 30.0),
                    "length": kwargs.get("length", 1000.0),
                    "glideslope": kwargs.get("glideslope", 3.0)
                }
            }
        elif task_type == "Landing":
            config['aircraft_config'][0]['task_config'] = {
                "type": "Landing",
                "config": {
                    "target_position": kwargs.get("target_position"),
                    "max_landing_speed": kwargs.get("max_speed", 25.0),
                    "max_descent_rate": kwargs.get("max_descent", 3.0),
                    "max_bank_angle": kwargs.get("max_bank", np.pi/6),
                    "max_landing_distance": kwargs.get("max_distance", 200.0),
                    "landing_complete_height": kwargs.get("complete_height", 0.5)
                }
            }

        self.env = self.create_env(config)

    def test_goal_termination(self):
        """Test goal reaching termination conditions."""
        # Test with different positions and tolerances
        test_cases = [
            {
                "position": {"x": 100.0, "y": 100.0, "z": -200.0},
                "tolerance": 10.0,
                "expected_term": False,
                "description": "Far from goal"
            },
            {
                "position": {"x": 5.0, "y": 5.0, "z": -100.0},
                "tolerance": 10.0,
                "expected_term": True,
                "description": "Within tolerance"
            }
        ]

        for case in test_cases:
            self.setup_task_config("Goal",
                                 position=case["position"],
                                 tolerance=case["tolerance"])
            obs, _ = self.env.reset()
            term = self.env.is_terminal()
            assert term == case["expected_term"], \
                f"For {case['description']}, expected termination={case['expected_term']}"

    def test_runway_termination(self):
        """Test runway task termination conditions."""
        test_cases = [
            {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "heading": 0.0,
                "width": 30.0,
                "expected_term": True,
                "description": "On threshold"
            },
            {
                "position": {"x": -100.0, "y": 0.0, "z": -50.0},
                "heading": 0.0,
                "width": 30.0,
                "expected_term": False,
                "description": "On approach"
            },
            {
                "position": {"x": 0.0, "y": 100.0, "z": -50.0},
                "heading": 0.0,
                "width": 30.0,
                "expected_term": True,
                "description": "Too far laterally"
            }
        ]

        for case in test_cases:
            self.setup_task_config("Runway",
                                 position=case["position"],
                                 heading=case["heading"],
                                 width=case["width"])
            obs, _ = self.env.reset()
            term = self.env.is_terminal()
            assert term == case["expected_term"], \
                f"For {case['description']}, expected termination={case['expected_term']}"

    def test_landing_termination(self):
        """Test landing task termination conditions."""
        test_cases = [
            {
                "max_speed": 25.0,
                "max_descent": 3.0,
                "max_bank": np.pi/6,
                "description": "Normal approach",
                "expected_term": False
            },
            {
                "max_speed": 25.0,
                "max_descent": 3.0,
                "max_bank": np.pi/6,
                "ground_contact": True,
                "description": "Ground contact",
                "expected_term": True
            },
            {
                "max_speed": 25.0,
                "max_descent": 3.0,
                "max_bank": np.pi/6,
                "excess_speed": True,
                "description": "Excessive speed",
                "expected_term": True
            }
        ]

        for case in test_cases:
            self.setup_task_config("Landing",
                                 max_speed=case["max_speed"],
                                 max_descent=case["max_descent"],
                                 max_bank=case["max_bank"])
            obs, _ = self.env.reset()
            # Simulate conditions if needed
            if case.get("ground_contact"):
                self.env.set_aircraft_height(0.0)
            if case.get("excess_speed"):
                self.env.set_aircraft_speed(case["max_speed"] * 2)

            term = self.env.is_terminal()
            assert term == case["expected_term"], \
                f"For {case['description']}, expected termination={case['expected_term']}"

    def test_trajectory_termination(self):
        """Test trajectory task termination conditions."""
        test_cases = [
            {
                "motion_type": {
                    "type": "StraightAndLevel",
                    "target_distance": 100.0
                },
                "distance_covered": 50.0,
                "expected_term": False,
                "description": "Mid-trajectory"
            },
            {
                "motion_type": {
                    "type": "StraightAndLevel",
                    "target_distance": 100.0
                },
                "distance_covered": 150.0,
                "expected_term": True,
                "description": "Beyond target distance"
            },
            {
                "motion_type": {
                    "type": "CoordinatedTurn",
                    "turn_radius": 100.0,
                    "turn_angle": 90.0,
                    "direction": "Right"
                },
                "turn_complete": True,
                "expected_term": True,
                "description": "Turn complete"
            }
        ]

        for case in test_cases:
            self.setup_task_config("Trajectory",
                                 motion_type=case["motion_type"],
                                 target_velocity=20.0)
            obs, _ = self.env.reset()
            # Simulate conditions
            if case.get("distance_covered"):
                self.env.set_aircraft_distance(case["distance_covered"])
            if case.get("turn_complete"):
                self.env.set_aircraft_heading(np.pi/2)  # 90 degree turn

            term = self.env.is_terminal()
            assert term == case["expected_term"], \
                f"For {case['description']}, expected termination={case['expected_term']}"
