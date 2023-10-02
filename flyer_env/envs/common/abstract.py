import os
from typing import List, Tuple, Optional, TypeVar, Dict, Text
import gymnasium as gym
import numpy as np

from flyer_env.envs.common.action import action_factory, Action, ActionType
from flyer_env.envs.common.observation import observation_factory, ObservationType

from pyflyer import Aircraft

Observation = TypeVar("Observation")

class AbstractEnv(gym.Env):

    """
    A generic environment for various aircraft flight related tasks
    
    """

    observation_type: ObservationType
    action_type: ActionType
    metadata = {
        'render_modes': ['human', 'rgb_array']
    }

    def __init__(
        self,
        config: dict = None,
        render_mode: Optional[str] = None,
    ) -> None:
        
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Scene
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0.0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
        self.reset()

    @property
    def vehicle(self) -> Aircraft:
        """First (default) controlled vehicle"""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None
    
    @vehicle.setter
    def vehicle(self, vehicle: Aircraft) -> None:
        """Set a unique controlled vehicle"""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration

        Can be overloaded within environment config or with configure()
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "Dynamics"
            },
            "action": {
                "type": "ContinuousAction"
            },
            "simulation_frequency": 120.0,  # [Hz]
            "policy_frequency": 1.0,  # [Hz]
            "render_frequency": 0.01,  # [Hz]
            "screen_size": 600,  # [px], forced to be square viewport for now
            "scaling": 25,  # [m/px], ratio of how large the default tile is in [m] 
        }

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        """
        Setup the types and spaces of observation from the config
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending in the given state

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError
    
    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return: True: if terminal, False: if not
        """
        raise NotImplementedError
    
    def _is_truncated(self) -> bool:
        """
        Check if the episode is truncated at the current step

        :return: True: if truncated, False: if not
        """
        raise NotImplementedError
    
    def _info(self, obs, action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """

        info = {
            #TODO: Add key information we might want here
        }

        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info
    
    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
        ) -> Tuple[Observation, dict]:
        """
        Reset the environment to it's initial configuration

        :param seed: The seed that is used to initialize the environment's PRNG
        :param options: Allows the environment configuration to specified through options["config"]
        :return: the observation of the reset state and information about the environment
        """

        super().reset(seed=seed, options=options)
        if options and "config" in options:
            self.configure(options["config"])
        
        # First, to set the controlled vehicle class depending on the action space
        self.define_spaces()
        
        self.time = 0.0
        self.steps = 0
        self.done = False
        self._reset(seed)

        # Second, to link the obs and actions to the vehicles once the scene is created
        self.define_spaces()
        obs = self.observation_type.observe()
        info = self._info(obs, action=self.action_space.sample())
        self.world.screen_dim = [self.config["screen_size"], self.config["screen_size"]]

        return obs, info

    def _reset(self) -> None:
        """
        Reset the scene

        Method overloaded by the environments
        """

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behavior
        for several simulation time-steps until the next decision-making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        
        # Call from the simulator and update the values of each
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, reward, terminated, truncated, info

    def _simulate(self, action) -> None:
        """
        Simulate the world
        """
        dt = 1/self.config["simulation_frequency"]
        self.time += dt
        print(f'action: {action}')
        self.action_type.act(action)
        self.vehicle.step(dt)  # Set the action on the aircraft
        # self.world.step()  # Step the world

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
                "You can specify the render_mode at initialization."
                f"e.g. gym.make({self.spec.id}, render_mode='rgb_array')"
            )
            return
        
        if self.render_mode == "rgb_array":
            bytes = self.world.render()
            img = np.array(bytes, dtype=np.uint8)
            img = img.reshape((int(self.world.screen_width), int(self.world.screen_height), 4))
            img = img[:, :, :3]

            return img
    
    def close(self) -> None:
        """
        Close the environment
        """
        self.done = True
        # TODO: Find a way to close the viewer if it exists

    def get_available_actions(self) -> List[int]:
        """
        Helper method to provide available actions in the current environment
        """
        return self.action_type.get_availale_actions()
