import gymnasium as gym
import pytest

from flyer_env.envs.flyer_env import FlyerEnv

envs = ["flyer-v1",
        "trajectory-v1",
        "runway-v1",
        "forced_landing-v1",
        "control-v1"]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
    env.close()

def test_env_reset_options(env_spec: str = "flyer-v1"):
    env = gym.make(env_spec)
    
    # Might want to add some more parameters to test here

    default_duration =  FlyerEnv().default_config()["duration"]
    assert env.unwrapped.config["duration"] == default_duration

    update_duration = default_duration * 2
    env.reset(options={"config": {"duration": update_duration}})
    assert env.unwrapped.config["duration"] == update_duration
