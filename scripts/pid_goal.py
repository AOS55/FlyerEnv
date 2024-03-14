from typing import Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from flyer_env import RecordVideo

# from gymnasium.wrappers.record_video import RecordVideo

plt.rcParams.update({"text.usetex": True})

COLOURS = [
    [0, 18, 25],
    [0, 95, 115],
    [10, 147, 150],
    [148, 210, 189],
    [233, 216, 166],
    [238, 155, 0],
    [202, 103, 2],
    [187, 62, 3],
    [174, 32, 18],
    [155, 34, 38],
]
COLOURS = [[value / 255 for value in rgb] for rgb in COLOURS]


def main():

    env_config = {
        "duration": 100.0,
        "action": {"type": "PursuitAction"},
        "simulation_frequency": 100.0,
    }

    env = gym.make("flyer-v1", config=env_config, render_mode="rgb_array")
    time = 0
    dt = 1 / env.unwrapped.config["simulation_frequency"]
    env = RecordVideo(env, "videos", dt, 1 / env.unwrapped.config["render_frequency"])
    # env = RecordVideo(env, "videos")

    obs, info = env.reset()
    observations = []
    times = []
    terminated = truncated = False
    goal = env.unwrapped.goal

    print(f"goal: {goal}")

    while not (terminated or truncated):
        action = {
            "goal_pos": goal[0:2],
            "other_controls": np.array([goal[-1] / 10000, 100.0 / 300.0]),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f'obs: {obs}')
        times.append(time)
        observations.append(obs)
    env.close()
    observations = np.array(observations)
    outputs = {
        "x": observations[:, 0, 0],
        "y": observations[:, 0, 1],
        "z": -1 * observations[:, 0, 2],
    }
    target = {
        "x": env.unwrapped.goal[0],
        "y": env.unwrapped.goal[1],
        "z": -1 * env.unwrapped.goal[2],
    }

    plot_position(outputs, target)
    plt.show()


def plot_position(outputs: Dict[str, float], target: Dict[str, float]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(outputs["x"], outputs["y"], outputs["z"], c=COLOURS[1])
    ax.scatter(target["x"], target["y"], target["z"], c=COLOURS[5])
    fig.show()


if __name__ == "__main__":
    main()
