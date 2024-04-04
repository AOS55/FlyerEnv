import os
from typing import Dict, Text

import numpy as np
from pyflyer import Aircraft, World

from flyer_env import utils
from flyer_env.aircraft import ControlledAircraft
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import Action


class ControlEnv(AbstractEnv):
    """
    A flying environment where the aircraft is tasked with maintaining a control based command.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Control"},
                "action": {"type": "ContinuousAction"},
                "area": (1024, 1024),
                "vechicle_type": "Dynamic",
                "duration": 10.0,
                "collision_reward": -200.0,
                "max_reward": 0.0,
                "normalize_reward": False,
                "state_com": {
                    "pitch": (0.0, 180.0 / np.pi),
                    "roll": (0.0, 180.0 / np.pi),
                    "yaw": (0.0, 180.0 / np.pi),
                    "u": (100.0, 0.1),
                },
            }
        )
        return config

    def _reset(self, seed=None) -> None:

        self.np_random = np.random.RandomState(seed)
        self._create_world()
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

    def _create_vehicles(self) -> None:
        """Create an aircraft to fly around the world"""
        self.controlled_vehicles = []
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
        start_pos = [0.0, 0.0, -1000.0]
        heading = 0.0
        airspeed = 100.0
        aircraft = Aircraft(data_path=path)
        aircraft.reset(pos=start_pos, heading=heading, airspeed=airspeed)
        self.world.add_aircraft(aircraft)

        if (
            self.config["action"]["type"] == "ContinuousAction"
            or self.config["action"]["type"] == "LongitudinalAction"
        ):
            vehicle = self.world.vehicles[0]
        else:
            vehicle = ControlledAircraft(
                self.world.vehicles[0],
                dt=1 / self.config["simulation_frequency"],
            )
        self.controlled_vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        Reward maximal when meeting specified conditions

        :param action: last action performed
        :return: reward
        """
        rewards = self._rewards(action)
        reward = sum(reward for _, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [-1000.0, 0.0], [0, 1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate

        :param action:
        :return:
        """
        crash_reward = self._crash_reward()
        state_reward = self._state_reward()
        return {"crash_reward": crash_reward, "state_reward": state_reward}

    def _state_reward(self) -> float:
        v_dict = self.controlled_vehicles[0].dict
        reward = 0.0
        for key in self.config["state_com"]:
            com, scale = self.config["state_com"][key]
            error = np.abs(com - v_dict[key])
            # print(f'{key}, {v_dict[key] * 180.0/np.pi}, error: {error}')
            reward -= error * scale
            # print(f'{key}, reward: {reward}')
        if reward < -1000.0:
            reward = -1000.0
            print(f"reward: {reward}")
        return reward

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
        The episode is over if the ego vehicle crashed, or it hits the ground
        """
        # If crashed terminate
        if self.vehicle.crashed:
            return True
        # If in ground terminate (not a landing scenario)
        if self.vehicle.position[-1] >= 0.0:
            return True
        return False

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached"""
        return self.time >= self.config["duration"]
