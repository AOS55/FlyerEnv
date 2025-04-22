# import os
from importlib.metadata import version

__version__ = version("flyer-env")

# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from gymnasium.envs.registration import register

# # from flyer_env.wrappers import RecordVideo

def register_flyer_envs():
    """Import the envs module so that the environs register themselves."""

    # control_env.py
    register(
        id="flyer_control-v1",
        entry_point="flyer_env.envs.single_agent.control:ControlFlyerEnv"
    )

    register(
        id="flyer_goal-v1",
        entry_point="flyer_env.envs.single_agent.goal:GoalFlyerEnv"
    )

    register(
        id="flyer_trajectory-v1",
        entry_point="flyer_env.envs.single_agent.trajectory:TrajectoryFlyerEnv"
    )
