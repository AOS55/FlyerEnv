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
    A goal-oriented flying environment

    The aircraft is tasked with navigating to a fixed point in space, in the shortest time possible, without crashing.
    Crashing occurs by colliding with the ground.
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
            "area": (256, 256),  # terrain map area [tiles]
            "vehicle_type": "Dynamic",  # vehicle type only dynamic
            "duration": 10.0,  # [s]
            "collision_reward": -200.0,  # max -ve reward for crashing
            "point_reward": 100.0,  # max +ve reward for hitting the goal
            "normalize_reward": True,
            "goal_generation": {
                "heading_limits": [-np.pi, np.pi],
                "pitch_limits": [-10.0 * np.pi/180.0, 10.0 * np.pi/180.0],
                "dist_limits": [1000.0, 10000.0],
                "dist_terminal": 20.0
            }
        })
        return config

    def _reset(self, seed) -> None:
        if not seed: seed = 1  # set seed to 1 if None TODO: set to be random on None, look @ HighwayEnv
        self._create_world(seed)
        self._create_vehicles()
        self._create_goal(seed)

    def _create_world(self, seed) -> None:
        """Create the world map"""
        self.world = World()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.world.assets_dir = path
        self.world.create_map(seed)
        return
    
    def _create_vehicles(self) -> None:
        """Create an aircraft to fly around the world"""
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

    
    def _create_goal(self, seed) -> None:
        """Create a random goal in 3D space to navigate to, based on the aircraft's initial starting position"""
        v_pos = self.world.vehicles[0].position
        gg = self.config["goal_generation"]

        np_random = np.random.RandomState(seed)

        def get_goal():
            heading = np_random.uniform(gg["heading_limits"][0], gg["heading_limits"][1])
            pitch = np_random.uniform(gg["pitch_limits"][0], gg["pitch_limits"][1])
            dist = np_random.uniform(gg["dist_limits"][0], gg["dist_limits"][1])
            rel_pos = dist*np.array([np.cos(pitch)*np.sin(heading), np.cos(pitch)*np.cos(heading), np.sin(pitch)])
            pos = v_pos + rel_pos
            return pos

        g_pos = get_goal()
        self.goal = g_pos
        return
    
    def _reward(self, action: Action) -> float:
        """
        Reward vehicle if it makes progress towards the goal state

        :param action: last action performed
        :return: reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                self.config["point_reward"]],
                                [0, 1])
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate point distance based rewards

        :param action: last action performed
        :return: dictionary of rewards
        """
        crash_reward = self._crash_reward()
        point_reward = self._point_reward()
        return {'collision_reward': crash_reward,
                'point_reward': point_reward}

    def _point_reward(self):
        """
        Reward for reaching the goal state
        """
        distance = self.controlled_vehicles[0].aircraft.goal_dist(self.goal)
        point_reward = self.config["point_reward"]
        dist_terminal = self.config["goal_generation"]["dist_terminal"]
        reward = point_reward * dist_terminal / distance
        return reward

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
