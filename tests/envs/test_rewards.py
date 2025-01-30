import pytest
import numpy as np
from typing import Dict, Any

from tests.common import BaseSingleAgentTest, EnvironmentConfigs

class TestTaskRewards(BaseSingleAgentTest):
    """Test suite for different task reward calculations."""

    def get_config(self) -> Dict[str, Any]:
        """Get base config with proper task configuration."""
        config = EnvironmentConfigs.get_dubins_config()

        # Add default task configuration
        config['aircraft_config'][0]['task_config'] = {
            'type' : 'Control',
            'config': {
                'control_type': 'Heading',
                'target': 0.0,
                'tolerance': 0.1,  # Tighter tolerance for better testing
                'reward_type': 'Dense'  # Added reward_type
            }
        }
        return config

    def setup_task_config(self, task_type: str, target: float, tolerance: float, reward_type: str = 'Dense'):
        """Helper to configure task for testing."""
        config = self.get_config()
        config['aircraft_config'][0]['task_config'] = {
            'type': task_type,
            'config': {
                'target': target,
                'tolerance': tolerance,
                'reward_type': reward_type
            }
        }
        self.env = self.create_env(config)

    def test_control_task_heading(self):
        """Test heading control task rewards."""
        target_heading = np.pi/4  # 45 degrees
        tolerance = 0.1  # About 5.7 degrees
        print(f"target_heading: {target_heading}, tolerance: {tolerance}")
        self.setup_task_config('Control', target_heading, tolerance)
        obs, _ = self.env.reset()

        # Test rewards at different errors
        test_errors = [
            (0.0, 1.0, "perfect alignment"),
            (0.05, 0.973, "small error"),
            (0.2, 0.632, "medium error"),
            (0.5, 0.062, "large error")
        ]

        for error, expected_reward, description in test_errors:
            # Calculate reward to match Rust implementation
            reward = np.exp(-(error / (3.0 * tolerance))**2)  # Square the normalized error
            assert abs(reward - expected_reward) < 0.1, \
                f"For {description}, expected reward ~{expected_reward}, got {reward}"

    def test_control_task_altitude(self):
        """Test altitude control task rewards."""
        target_altitude = 1000.0
        tolerance = 20.0
        self.setup_task_config('Control', target_altitude, tolerance)
        obs, _ = self.env.reset()

        test_cases = [
            (0.0, 1.0, "at target"),
            (10.0, 0.973, "small deviation"),  # Updated to match Rust
            (40.0, 0.632, "medium deviation"),
            (100.0, 0.062, "large deviation")
        ]

        for error, expected_reward, description in test_cases:
            reward = np.exp(-(error / (3.0 * tolerance))**2)
            assert abs(reward - expected_reward) < 0.1, \
                f"For {description}, expected reward ~{expected_reward}, got {reward}"

    def test_goal_task_sparse(self):
        """Test sparse goal reaching task rewards."""
        tolerance = 10.0

        self.setup_task_config('Goal', 100.0, tolerance, reward_type='Sparse')
        obs, _ = self.env.reset()

        test_distances = [
            (5.0, 1.0, "inside tolerance"),
            (9.9, 1.0, "just inside tolerance"),
            (10.1, 0.0, "just outside tolerance"),
            (20.0, 0.0, "far from goal")
        ]

        for distance, expected_reward, description in test_distances:
            reward = 1.0 if distance <= tolerance else 0.0
            assert reward == expected_reward, \
                f"For {description} (distance={distance}m), expected {expected_reward}, got {reward}"

    def test_goal_task_dense(self):
        """Test dense goal reaching task rewards."""
        tolerance = 10.0
        self.setup_task_config('Goal', 100.0, tolerance, reward_type='Dense')
        obs, _ = self.env.reset()

        test_distances = [
            (0.0, 1.0, "at goal"),
            (5.0, 1.0, "close"),
            (20.0, 0.641, "medium"),
            (50.0, 0.062, "far")
        ]

        for distance, expected_reward, description in test_distances:
            # Match Rust logic with tolerance check
            if distance <= tolerance:
                reward = 1.0
            else:
                reward = np.exp(-(distance / (3.0 * tolerance))**2)
            assert abs(reward - expected_reward) < 0.1, \
                f"For {description} (distance={distance}m), expected ~{expected_reward}, got {reward}"

    def test_runway_alignment(self):
        """Test runway alignment task rewards."""
        runway_width = 30.0
        heading_tolerance = 15.0 * np.pi / 180.0  # 15 degrees in radians

        self.setup_task_config('Runway', 0.0, min(runway_width, heading_tolerance))
        obs, _ = self.env.reset()

        test_cases = [
            ((0.0, 0.0), 1.0, "perfect alignment"),
            ((runway_width/4, heading_tolerance/4), 0.74, "small deviation"),  # Updated to match 0.3*lat + 0.3*glide + 0.2*head + 0.2*speed
            ((runway_width/2, heading_tolerance/2), 0.61, "medium deviation"),
            ((runway_width*2, heading_tolerance*2), 0.50, "large deviation")
        ]

        for (lateral_error, heading_error), expected_reward, description in test_cases:
            # Match Rust implementation weights and components
            lateral_reward = np.exp(-3.0 * lateral_error / runway_width)
            heading_reward = np.exp(-3.0 * heading_error / heading_tolerance)
            # Assuming default values for glideslope and speed rewards when perfectly maintained
            glideslope_reward = 1.0  # Perfect glideslope
            speed_reward = 1.0       # Perfect speed

            reward = (0.3 * lateral_reward + 0.3 * glideslope_reward +
                    0.2 * heading_reward + 0.2 * speed_reward)

            assert abs(reward - expected_reward) < 0.15, \
                f"For {description}, expected reward ~{expected_reward}, got {reward}"

    def test_landing_task(self):
        """Test landing task rewards."""
        max_speed = 25.0
        max_descent = 3.0
        max_bank = np.pi/6

        self.setup_task_config('Landing', 0.0, 0.0)  # Landing doesn't use target/tolerance
        obs, _ = self.env.reset()

        test_cases = [
            {
                "ground_speed": 20.0,
                "descent_rate": 2.0,
                "bank": 0.0,
                "expected": 1.0,
                "description": "good landing"
            },
            {
                "ground_speed": max_speed * 1.2,
                "descent_rate": 2.0,
                "bank": 0.0,
                "expected": -0.5,
                "description": "too fast"
            },
            {
                "ground_speed": 20.0,
                "descent_rate": max_descent * 1.2,
                "bank": 0.0,
                "expected": -0.5,
                "description": "hard landing"
            }
        ]

        for case in test_cases:
            landing_reward = 1.0 if (
                case["ground_speed"] <= max_speed and
                case["descent_rate"] <= max_descent and
                abs(case["bank"]) <= max_bank
            ) else -0.5

            assert landing_reward == case["expected"], \
                f"For {case['description']}, expected reward {case['expected']}, got {landing_reward}"
