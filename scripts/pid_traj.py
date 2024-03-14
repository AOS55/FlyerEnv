import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict
from flyer_env import utils

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
        "duration": 10.0,
        "observation": {"type": "Trajectory"},
        "action": {"type": "PursuitAction"},
        "simulation_frequency": 1000.0,
        "trajectory_config": {"name": "sl"},
    }

    env = gym.make("trajectory-v1", config=env_config)
    time = 0
    dt = 1 / env.unwrapped.config["simulation_frequency"]

    obs, info = env.reset()
    observations = []
    times = []
    terminated = truncated = False
    targets = []
    speed = utils.lmap(
        env.unwrapped.vehicle.dict["u"],
        env.unwrapped.action_type.speed_range,
        [0.0, 1.0],
    )
    alt = utils.lmap(info["t_pos"][-1], env.unwrapped.action_type.alt_range, [0.0, 1.0])

    while not (terminated or truncated):
        action = {
            "goal_pos": np.array([info["t_pos"][0], info["t_pos"][1]]),
            "other_controls": np.array([alt, speed]),
        }

        v_dict = env.unwrapped.vehicle.dict
        controls = env.unwrapped.vehicle.aircraft.controls

        obs, reward, terminated, truncated, info = env.step(action)
        obs_dict = {
            "elevator": controls["elevator"],
            "aileron": controls["aileron"],
            "rudder": controls["rudder"],
            "x": env.unwrapped.vehicle.dict["x"],
            "y": env.unwrapped.vehicle.dict["y"],
            "z": env.unwrapped.vehicle.dict["z"],
            "x_com": info["t_pos"][0],
            "y_com": info["t_pos"][1],
            "z_com": info["t_pos"][-1],
            "pitch": env.unwrapped.vehicle.dict["pitch"],
            "roll": env.unwrapped.vehicle.dict["roll"],
            "yaw": env.unwrapped.vehicle.dict["yaw"],
            "u": env.unwrapped.vehicle.dict["u"],
            "reward": reward,
        }
        times.append(time)
        targets.append(info["t_pos"])
        observations.append(obs_dict)
        time += dt
    env.close()
    observations = pd.DataFrame.from_dict(observations)
    plot_long(observations, times, env_config["duration"])
    plot_lat(observations, times, env_config["duration"])
    plot_track(observations)
    plt.show()


def plot_long(outputs, times, exp_len):
    fig, ax = plt.subplots(5, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Longitudinal Tracking}")
    ax[0].plot(times, outputs["elevator"], c=COLOURS[5], label=r"elevator")
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r"\textbf{Control}")

    ax[1].plot(times, outputs["pitch"] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[2].plot(times, outputs["z"], c=COLOURS[1])
    ax[2].plot(times, outputs["z_com"], linestyle="dashed", c=COLOURS[2])
    ax[2].set_ylabel(r"$z [m]$", fontsize=15)

    ax[3].plot(times, outputs["u"], c=COLOURS[1])
    ax[3].set_ylabel(r"$u [\frac{m}{s}]$", fontsize=15)

    ax[4].plot(times, outputs["reward"], c=COLOURS[1])
    ax[4].set_ylabel(r"Reward", fontsize=15)
    ax[4].set_xlabel(r"time [$s$]", fontsize=15)

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.show()


def plot_lat(outputs, times, exp_len):

    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Lateral-Directional Tracking}")
    ax[0].plot(times, outputs["aileron"], c=COLOURS[5], label=r"aileron")
    ax[0].plot(
        times, outputs["rudder"], c=COLOURS[7], linestyle="dashed", label=r"rudder"
    )
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r"\textbf{Control}")

    ax[1].plot(times, outputs["roll"] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].plot(
        times,
        outputs["yaw"] * 180.0 / np.pi,
        c=COLOURS[2],
        linestyle="dashed",
        label=r"$\psi$",
    )
    ax[1].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[2].plot(times, outputs["u"], c=COLOURS[1])
    ax[2].set_ylabel(r"$u [\frac{m}{s}]$", fontsize=15)

    ax[3].plot(times, outputs["reward"], c=COLOURS[1])
    ax[3].set_ylabel(r"Reward", fontsize=15)
    ax[3].set_xlabel(r"time [$s$]", fontsize=15)

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.show()


def plot_track(outputs):

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.plot(outputs["x"], outputs["y"], c=COLOURS[1])
    ax.plot(outputs["x_com"], outputs["y_com"], linestyle="dashed", c=COLOURS[2])
    ax.set_ylabel(r"$y [m]$", fontsize=15)
    ax.set_xlabel(r"$x [m]$", fontsize=15)
    ax.set_aspect("equal")
    ax.grid()

    fig.show()


if __name__ == "__main__":
    main()
