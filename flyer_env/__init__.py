from importlib.metadata import version

__version__ = version("flyer-env")

from gymnasium.envs.registration import register

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

    # register(
    #     id="flyer_runway-v1",
    #     entry_point="flyer_env.envs.single_agent.runway:RunwayFlyerEnv"
    # )

    # register(
    #     id="flyer_landing-v1",
    #     entry_point="flyer_env.envs.single_agent.landing:LandingFlyerEnv"
    # )
