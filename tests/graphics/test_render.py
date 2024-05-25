import gymnasium as gym
import numpy as np
import pytest

envs = ["flyer-v1",
        "trajectory-v1",
        "runway-v1",
        "forced_landing-v1",
        "control-v1"]


@pytest.mark.parametrize("env_spec", envs)
def test_render(env_spec):
    env = gym.make(env_spec, render_mode="rgb_array")
    env.unwrapped.configure({"offscreen_rendering": True})
    env.reset()
    img = env.render()
    print(f'type(image): {type(img)}')
    env.close()
    assert isinstance(img, np.ndarray)
    assert img.shape == (
        env.unwrapped.config["screen_size"],
        env.unwrapped.config["screen_size"],
        3
    )  # (H,W,C)

