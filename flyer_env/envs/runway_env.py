from typing import Dict, Text
import os
import numpy as np
from flyer_env import utils
from flyer_env.aircraft import ControlledAircraft
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import Action
from pyflyer import World, Aircraft


class RunwayEnv(AbstractEnv):
    """
    A runway landing environment

    The aircraft is tasked with safely landing on a runway without crashing.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Dynamics"},
                "action": {"type": "ContinuousAction"},
                "area": (1024, 1024),  # terrain map area [tiles]
                "vehicle_type": "Dynamic",  # vehicle type, only dynamic available
                "duration": 10.0,  # simulation duration [s]
                "collision_reward": -200.0,  # max -ve reward for crashing
                "landing_reward": 100.0,  # max +ve reward for landing successfully
                "normalize_reward": True,  # whether to normalize the reward
                "runway_configuration": {
                    "runway_position": [0.0, 0.0],
                    "runway_width": 20.0,
                    "runway_length": 1500.0,
                    "runway_heading": 0.0,
                },  # trajectory configuration details
            }
        )
        return config

    def _reset(self) -> None:
        self.np_random = np.random.RandomState()
        self._create_world()
        self._create_runway()
        self._create_vehicles()

    def _create_world(self) -> None:
        """Create the world map"""
        self.world = World()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.world.assets_dir = path
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "terrain_data")
        self.world.terrain_data_dir = path
        world_seed = self.np_random.randint(100)  # set 100 possible seeds by default
        self.world.create_map(world_seed, area=self.config["area"])
        return

    def _create_runway(self) -> None:
        """Create a runway for the aircraft to land on"""
        runway_config = self.config["runway_configuration"]
        self.world.add_runway(
            runway_position=runway_config["runway_position"],
            runway_width=runway_config["runway_width"],
            runway_heading=runway_config["runway_heading"],
        )

    def _create_vehicles(self) -> None:
        """Create an aircaft to fly around the world"""
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
        start_pos = [0.0, 0.0, -1000.0]
        heading = 0.0
        airspeed = 100.0
        aircraft = Aircraft(data_path=path)
        aircraft.reset(pos=start_pos, heading=heading, airspeed=airspeed)
        self.world.add_aircraft(aircraft)

        # Due to borrow checker of World.rs need to build from this reference
        if self.config["action"]["type"] == "ContinuousAction":
            vehicle = self.world.vehicles[0]
        else:
            vehicle = ControlledAircraft(
                self.world.vehicles[0],
                dt=1 / self.config["simulation_frequency"],
            )
        self.controlled_vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """Reward vehicle if it lands successfully"""
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["landing_reward"]],
                [0, 1],
            )
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate landing based rewards
        """
        crash_reward = self._crash_reward()
        landing_reward = self._landing_reward()
        return {"collision_reward": crash_reward, "landing_reward": landing_reward}

    def _landing_reward(self):
        """
        Reward for landing successfully
        """
        v_pos = self.vehicle.position
        if v_pos[-1] > -4 and self.world.point_on_runway(v_pos[0:2]):
            return self.config["landing_reward"]
        return 0.0

    def _crash_reward(self) -> float:
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
