from typing import Dict, Text
import os
import numpy as np
from flyer_env import utils
from flyer_env.aircraft import ControlledAircraft
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import Action
from pyflyer import World, Aircraft


class ForcedLandingEnv(AbstractEnv):    

    """
    A forced landing environment

    The aircraft is tasked with finding a suitable landing spot
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
            "duration": 500.0,  # simulation duration [s]
            "collision_reward": -200.0,  # max -ve reward for crashing
            "landing_reward": 100.0,  # max +ve reward for landing successfully
            "normalize_reward": True  
        })

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
        return
    
    def _create_vehicles(self) -> None:
        """Create an aircaft to fly around the world"""
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
        start_pos = [0.0, 0.0, -1000.0]
        heading = 0.0
        airspeed = 100.0
        aircraft = Aircraft(data_path=path)
        aircraft.reset(
            pos=start_pos,
            heading=heading,
            airspeed=airspeed
        )
        self.world.add_aircraft(aircraft)

        # Due to borrow checker of World.rs need to build from this reference
        if self.config["action"]["type"] == "ContinuousAction":
            vehicle = self.world.vehicles[0]
        else:
            vehicle = ControlledAircraft(
                self.world.vehicles[0],
                dt = 1/self.config["simulation_frequency"],
            )
        self.controlled_vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """Reward vehicle if it lands successfully"""
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["landing_reward"]],
                                 [0, 1])
        return reward
    
    def _landing_reward(self):
        """
        Reward for successfully landing without crashing
        """
        return 
    
    def _crash_reward(self):
        """
        Penalize if the aircraft crashes
        """
        if self.vehicle.crashed:
            return self.config["collision_reward"]
        else:
            return 0.0

    def _is_terminated(self) -> bool:
        """
        The episode is over if the the ego vehicle crashed, or it hits the ground
        """

        v_pos = self.vehicle.position

        # If crashed terminate
        if self.vehicle.crashed:
            print("Crashed!")
            return True
        # If landed
        if v_pos[-1] > -4 and self.world.point_on_runway(v_pos[0:2]):
            print("Landed!")
            return True
        return False
    
    def _is_truncated(self) -> bool:
        """
        Episode is truncated if the time limit is reached
        """
        return self.time >= self.config["duration"]