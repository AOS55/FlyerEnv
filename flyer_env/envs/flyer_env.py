import os
from abc import abstractmethod
from typing import Dict, Text

import numpy as np
from gymnasium import Env
from pyflyer import Aircraft, World

from flyer_env import utils
from flyer_env.aircraft import ControlledAircraft
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import Action


class GoalEnv(Env):
    """
    Interface for a goal-based environment

    Similar to HighwayEnv https://github.com/Farama-Foundation/HighwayEnv/blob/master/highway_env/envs/parking_env.py.
    This interface is needed for agents to interact with agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.

    As a goal-based environment it functions in the same way as any regular OpenAI Gym Environment, but imposes a required structure on the obs space.
    More concretely, the observation space is required to contain at least 3 elements, namely `observation`, `desired_goal`, and `achieved goal`.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """
        Compute the step reward. This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved.

        :param achieved_goal: the goal that was achieved during execution
        :param desired_goal: the desired goal that we asked the agent to attempt to achieve
        :param info (dict): an info dictionary with additional information
        :return: the reward the corresponds to the provided goal achieved w.r.t. the desired goal

        """
        raise NotImplementedError


class FlyerEnv(AbstractEnv, GoalEnv):
    """
    A goal-oriented flying environment

    The aircraft is tasked with navigating to a fixed point in space, in the shortest time possible, without crashing.
    Crashing occurs by colliding with the ground.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Goal"},
                "action": {"type": "ContinuousAction"},
                "area": (1024, 1024),  # terrain map area [tiles]
                "vehicle_type": "Dynamic",  # vehicle type, only dynamic available
                "duration": 10.0,  # simulation duration [s]
                "collision_reward": -100.0,  # max -ve reward for crashing
                "reward_type": "dense",  # reward type
                "point_reward": 1.0,  # multiplier for distance from goal
                "normalize_reward": False,  # whether to normalize the reward [-1, +1], not working at the moment
                "goal_generation": {
                    "heading_limits": [85.0 * np.pi / 180.0, 95.0 * np.pi / 180.0],
                    "pitch_limits": [-0.1 * np.pi / 180.0, 0.1 * np.pi / 180.0],
                    "dist_limits": [1000.0, 10000.0],
                    "dist_terminal": 100.0,
                },  # goal generation details
            }
        )
        return config

    def _info(self, obs, action) -> dict:
        info = super(FlyerEnv, self)._info(obs, action)
        success = self._is_success()
        info.update({"is_success": success})
        return info

    def _reset(self, seed=None) -> None:

        self.np_random = np.random.RandomState(seed)
        self._create_world()
        self._create_vehicles()
        self._create_goal()

    def _create_world(self) -> None:
        """Create the world map"""
        self.world = World()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
        self.world.assets_dir = path
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "terrain_data")
        self.world.terrain_data_dir = path
        world_seed = self.np_random.randint(100)  # set 100 possible seeds by default
        self.world.create_map(world_seed, area=self.config["area"])
        self.world.render_type = "aircraft"

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

        # Due to borrow checker of World.rs need to build from this reference
        if self.config["action"]["type"] == "ContinuousAction":
            vehicle = self.world.vehicles[0]
        else:
            vehicle = ControlledAircraft(
                self.world.vehicles[0],
                dt=1 / self.config["simulation_frequency"],
            )
        self.controlled_vehicles.append(vehicle)

    def _create_goal(self) -> None:
        """Create a random goal in 3D space to navigate to, based on the aircraft's initial starting position"""
        v_pos = self.world.vehicles[0].position
        gg = self.config["goal_generation"]

        def get_goal():
            heading = self.np_random.uniform(
                gg["heading_limits"][0], gg["heading_limits"][1]
            )
            pitch = self.np_random.uniform(gg["pitch_limits"][0], gg["pitch_limits"][1])
            dist = self.np_random.uniform(gg["dist_limits"][0], gg["dist_limits"][1])
            rel_pos = dist * np.array(
                [
                    np.cos(pitch) * np.sin(heading),
                    np.cos(pitch) * np.cos(heading),
                    np.sin(pitch),
                ]
            )
            # print(f'rel_pos: {rel_pos}, pitch: {pitch}, heading: {heading}, dist: {dist}')
            pos = v_pos + rel_pos
            return pos

        g_pos = get_goal()
        self.goal = g_pos
        return

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """
        Proximity to goal is rewarded
        Just use _point_reward for now, could be more explicit.
        TODO: look at how the gripper point robots select the points
        """

        def _goal_distance(goal_a, goal_b):
            assert goal_a.shape == goal_b.shape
            return np.linalg.norm(goal_a - goal_b, axis=-1)

        dist_terminal = self.config["goal_generation"]["dist_terminal"]
        d = _goal_distance(achieved_goal, desired_goal)
        # print(f'distance: {d}, achieved_goal: {achieved_goal}, desired_goal: {desired_goal}')
        if self.config["reward_type"] == "sparse":
            return -(d > dist_terminal).astype(np.float32)
        else:
            return (
                -d
                * 100.0
                / (
                    self.config["goal_generation"]["dist_limits"][1]
                    * self.config["duration"]
                    * self.config["simulation_frequency"]
                )
            )

    def _reward(self, action: Action) -> float:
        """
        Reward vehicle if it makes progress towards the goal state

        :param action: last action performed
        :return: reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [self.config["collision_reward"], self.config["point_reward"]],
                [0, 1],
            )
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate point distance based rewards

        :param action: last action performed
        :return: dictionary of rewards
        """
        crash_reward = self._crash_reward()
        point_reward = self._point_reward()
        return {"collision_reward": crash_reward, "point_reward": point_reward}

    def _point_reward(self):
        """
        Reward for reaching the goal state
        """
        distance = self.vehicle.goal_dist(self.goal)
        # point_reward = self.config["point_reward"]
        # dist_terminal = self.config["goal_generation"]["dist_terminal"]
        # reward = point_reward * dist_terminal / distance
        # print(f'distance: {distance}, self.goal: {self.goal}, pos: {self.vehicle.dict}')
        reward = (
            -distance
            * 100.0
            / (
                self.config["goal_generation"]["dist_limits"][1]
                * self.config["duration"]
                * self.config["simulation_frequency"]
            )
        )
        return reward

    def _crash_reward(self) -> float:
        """
        Penalize if the aircraft crashes
        """
        if self.vehicle.crashed:
            return 1.0
        else:
            return 0.0

    def _is_success(self) -> bool:
        v_pos = self.vehicle.position
        difference = np.subtract(v_pos, self.goal)
        distance = np.linalg.norm(difference)
        dist_terminal = self.config["goal_generation"]["dist_terminal"]
        return distance < dist_terminal

    def _is_terminated(self) -> bool:
        """
        The episode is over if the the ego vehicle crashed, or it hits the ground
        """

        # v_pos = self.vehicle.position
        # difference = np.subtract(v_pos, self.goal)
        # distance = np.linalg.norm(difference)
        # dist_terminal = self.config["goal_generation"]["dist_terminal"]

        # If crashed terminate
        if self.vehicle.crashed:
            return True
        # If in ground terminate (not a landing scenario)
        if self.vehicle.position[-1] >= 0.0:
            return True
        # If reached goal region
        if self._is_success():
            return True
        return False

    def _is_truncated(self) -> bool:
        """
        Episode is truncated if the time limit is reached
        """
        return self.time >= self.config["duration"]
