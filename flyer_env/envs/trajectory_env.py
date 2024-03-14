import os
import numpy as np
from typing import Dict, Text

from flyer_env import utils
from flyer_env.envs.common.action import Action
from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.aircraft import ControlledAircraft
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
            "area": (1024, 1024),  # terrain map area [tiles]
            "vehicle_type": "Dynamic",  # vehicle type, only dynamic available
            "duration": 10.0,  # simulation duration [s]
            "collision_reward": -200.0,  # max -ve reward for crashing
            "traj_reward": 10.0,  # max +ve reward for reaching for following trajectory
            "normalize_reward": False,  # whether to normalize the reward [-1, +1]
            "start_displacement": 100.0, # offset distance from goal
            "trajectory_config": {
                "name": "climb",  
                "final_height": 200.0,
                "climb_angle": 10.0 * np.pi / 180.0,
                "length": 15.0,
            }  # trajectory configuration details
        })
        return config
    
    def _reset(self, seed) -> None:
        if not seed: seed = 1
        self._create_world(seed)
        self._create_vehicles()
        self._create_trajectory_func()

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
        """Create an aircraft to fly around the world"""
        self.controlled_vehicles = []
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

    def _create_trajectory_func(self):
        """Create a trajectory function for the aicraft to follow"""
        def traj_start_pos(ac_pos, ac_hdg, dist):
            ac_pos[0] = ac_pos[0] + dist * np.cos(ac_hdg)
            ac_pos[1] = ac_pos[1] + dist * np.sin(ac_hdg + np.pi/180.0)
            return ac_pos

        trajectory_config = self.config["trajectory_config"]
        v = self.world.vehicles[0].dict
        ac_pos = [v['x'], v['y'], v['z']]
        ac_hdg = v['yaw']
        # start_displacement = 10.0  # distance to displac the target from the aircraft start position
        
        start_pos = traj_start_pos(ac_pos, ac_hdg, self.config["start_displacement"])

        self.goal = start_pos
        self.traj_target = TrajectoryTarget(speed=v['u'],
                                            start_position=start_pos,
                                            start_heading=v['yaw'])
        traj_funcs = {"sl": self.traj_target.straight_and_level,
                      "climb": self.traj_target.climb,
                      "descend": self.traj_target.descend,
                      "lt": self.traj_target.left_turn,
                      "rt": self.traj_target.right_turn}
        func = traj_funcs[trajectory_config["name"]]
        self.traj_func = func(**trajectory_config)

    def _reward(self, action: Action) -> float:
        """
        Reward maximal when following the desired trajectory, controlled by a target ball.

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
        dt = 1/self.config["simulation_frequency"]
        # traj_reward = self.config["traj_reward"]
        t_pos, done = self.traj_target.update(self.traj_func, dt=dt)
        dist = self.controlled_vehicles[0].goal_dist(t_pos)
        self.goal = t_pos
        dist = dist-self.config["start_displacement"]
        if np.abs(dist) < 1.0:
            reward = 1.0
        else:
            reward = 1.0 / (dist**2)
        return reward

    def _crash_reward(self) -> float:
        """
        Penalize if the aircraft crashes
        """
        if self.vehicle.crashed: 
            return self.config["collision_reward"]
        else: 
            return 0.0

    def _info(self, obs, action) -> Dict[str, float]:
        """
        Dictionary for the trajectory target
        """
        return {"t_pos": self.goal}

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
