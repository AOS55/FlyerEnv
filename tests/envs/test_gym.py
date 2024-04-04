import gymnasium as gym
import pytest

import flyer_env

# flyer_env.register_flyer_envs()

envs = ["flyer-v1",
        "trajectory-v1",
        "runway-v1",
        "forced_landing-v1",
        "control-v1"]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):
    env = gym.make(env_spec)

    obs, info = env.reset()
    assert env.observation_space.contains(obs)

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
    env.close()

def test_env_reset_options(env_spec: str = "flyer-v1"):
    env = gym.make(env_spec)
    # TODO: test all env reset options
