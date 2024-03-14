import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from gymnasium.envs.registration import register
from flyer_env.wrappers import RecordVideo


def register_flyer_envs():
    """Import the envs module so that the environs register themselves."""

    # flyer_env.py
    register(id="flyer-v1", entry_point="flyer_env.envs:FlyerEnv")

    # trajectory_env.py
    register(id="trajectory-v1", entry_point="flyer_env.envs:TrajectoryEnv")

    # runway_env.py
    register(id="runway-v1", entry_point="flyer_env.envs:RunwayEnv")

    # forced_landing_env.py
    register(id="forced_landing-v1", entry_point="flyer_env.envs:ForcedLandingEnv")

    # control_env.py
    register(id="control-v1", entry_point="flyer_env.envs:ControlEnv")
