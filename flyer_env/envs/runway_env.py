from typing import Dict, Text
import sys
import os
import numpy as np
from flyer_env import utils
from flyer_env.aircraft import ControlledAircraft
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import Action
from pyflyer import World, Aircraft


class FlyerEnv(AbstractEnv):
    """
    A runway landing environment
    
    The aircraft is tasked with safely landing on a runway without crashing.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Dynamics"
            },
            "action": {
                "type": "ContinuousAction"
            },
            "area": (1024, 1024),  # terrain map area [tiles]
            "vehicle_type": "Dynamic",  # vehicle type, only dynamic available
            "duration": 10.0,  # [s]
            "collision_reward": -200.0,  # max -ve reward for crashing
            "landing_reward": 100.0,  # max +ve reward for landing successfully
            "normalize_reward": True,
            "runway_configuration": {
                "runway_position": [0.0, 0.0],
                "runway_width": 20.0,
                "runway_length": 1500.0,
                "runway_heading": 0.0
            }
        })
        return config
    
    def _reset(self, seed) -> None:
        if not seed: seed = 1
        self._create_world(seed)
        self._create_vehicles()

    def _create_world(self, seed) -> None:
        """Create the world map"""
        self.world = World()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.world.assets_dir = path
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "terrain_data")
        self.world.terrain_data_dir = path
        self.world.create_map(seed, area=self.config["area"])
        runway_config = self.config["runway_configuration"]
        self.world.create_runway(
            runway_position = runway_config["runway_position"],
            runway_width = runway_config["runway_width"],
            runway_heading = runway_config["runway_heading"]
        )
        return
    
    def _create_vehicles(self) -> None:
        """Create an aircaft to fly around the world"""

    def _reward(self, action: Action) -> float:
        """Reward vehicle if it lands successfully"""

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate landing based rewards
        """
        crash_reward = self._crash_reward()
        landing_reward = self._point_reward()
        return {'collision_reward': crash_reward,
                'landing_reward': landing_reward}

    def _landing_reward(self):
        """
        Reward for landing successfully 
        """

    def _crash_reward(self) -> float:
        """
        Penalize if the aircraft crashes
        """
        if self.vehicle.crashed:
            return -200.0
        else:
            return 0.0
    
    def _is_terminated(self) -> bool:
        """
        The episode is over if the the ego vehicle crashed, or it hits the ground
        """

        v_pos = self.vehicle.position
        difference = np.subtract(v_pos, self.goal)
        distance = np.linalg.norm(difference)
        dist_terminal = self.config["goal_generation"]["dist_terminal"]

        # If crashed terminate
        if self.vehicle.crashed:
            return True
        # If in ground terminate (not a landing scenario)
        if self.vehicle.position[-1] >= 0.0:
            return True
        # If reached goal region
        if distance < dist_terminal:
            return True
        return False

    def _is_truncated(self) -> bool:
        """
        Episode is truncated if the time limit is reached
        """
        return self.time >= self.config["duration"]
    