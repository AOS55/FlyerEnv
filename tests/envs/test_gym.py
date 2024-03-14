import gymnasium as gym
import pytest

import flyer_env

flyer_env.register_flyer_envs()

envs = ["flyer-v1"]


@pytest.mark.parametrize("env_spec", envs)
def test_env_step(env_spec):

    env_configuration = {"screen_size": 256, "duration": 10.0}
    env = gym.make(env_spec, config=env_configuration, render_mode="rgb_array")
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    observations = []
    terminated = truncated = False

    img = env.render()
    assert img.shape == (
        int(env.unwrapped.world.screen_width),
        int(env.unwrapped.world.screen_height),
        3,
    )

    ids = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs[0])
        assert env.observation_space.contains(obs)
        ids += 1
    env.close()


if __name__ == "__main__":
    test_env_step("flyer-v1")
