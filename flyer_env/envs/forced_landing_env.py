import numpy as np
from typing import Dict, Text

from flyer_env import utils
from flyer_env.envs.common.action import Action
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.aircraft.trajectory import TrajectoryTarget

from pyflyer import World, Aircraft

class TrajectoryEnv(AbstractEnv):

    """
    A flying environment where the aircraft is tasked with maintaining a track along a fixed trajectory without crashing. Crashing occurs if the aircraft collides with the ground.
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
            "area": (256, 256),
            "vehicle_type": "Dynamic",
            "duration": 10.0,
            "collision_reward": -200.0,
            "traj_reward": 10.0,
            "normalize_reward": True,
            "trajectory_config": {
                "name": "climb",
                "final_height": 200.0,
                "climb_angle": 10.0 * np.pi / 180.0,
                "length": 15.0,
            }
        })
        return config
    
    def _reset(self) -> None:
        self._create_world()
        self._create_aircraft()
        self._create_trajectory_func()

    def _create_world(self, seed) -> None:
        """Create the world map"""
        self.world = World()
        self.world.create_map(seed)
        return
    
    def _create_aircraft(self) -> None:
        """Create an aircraft to fly around the world"""
        aircraft = Aircraft()
        aircraft.reset(
            pos=[0.0, 0.0, -1000.0],
            heading = 0.0,
            airspeed=100.0
        )
        self.world.add_aircraft(aircraft)
        self.controlled_vehicles = self.world.vehicles

    def _create_trajectory_func(self):
        trajectory_config = self.config["trajectory_config"]
        v = self.world.vehciles[0]
        self.traj_target = TrajectoryTarget(speed=v.speed,
                                            start_position=v.position,
                                            start_heading=heading)
        traj_funcs = {"sl": self.traj_target.straight_and_level,
                      "climb": self.traj_target.climb,
                      "descend": self.traj_target.descend,
                      "lt": self.traj_target.left_turn,
                      "rt": self.traj_target.right_turn}
        func = traj_funcs[trajectory_config["name"]]
        self.traj_func = func(**trajectory_config)

    def _reward(self, action: Action) -> float:
        """
        Reward maximial when following the desired trajectory, controlled by a target ball.

        :param action: last action performed
        :return: reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["traj_reward"]],
                                [0, 1])

        return reward
    
    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Calculate

        :param action:
        :return:
        """
        crash_reward = self._crash_reward()
        traj_reward = self._traj_reward()
        return {
            'crash_reward': crash_reward,
            'traj_reward': traj_reward
        }
    
    def _traj_reward(self) -> float:
        dt = 1/self.config["policy_frequency"]
        traj_reward = self.config["traj_reward"]
        t_pos, done = self.traj_target.update(self.traj_func, dt=dt)
        v_pos = self.vehicle.position
        difference = np.subtract(v_pos, t_pos)
        dist = np.linalg.norm(difference)
        if dist < 1.0:
            reward = traj_reward
        else:
            reward = traj_reward / dist
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
        The episode is over if the ego vehicle crashed, or it hits the ground
        """
        # If crashed terminate
        if self.vehicle.crashed:
            return True
        # If in ground terminate (not a landing scenario)
        if self.vehicle.position[-1] <= 0.0:
            return True
        return False
    
    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached"""
        return self.time >= self.config["duration"]
