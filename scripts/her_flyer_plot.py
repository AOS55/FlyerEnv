import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

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
        "observation": {"type": "LateralGoal"},
        "action": {"type": "HeadingAction", "heading_range": (-0.1, 0.1)},
        "duration": 100.0,
        "simulation_frequency": 100.0,
        "seed": 0,
        "goal_generation": {
            "heading_limits": [85.0 * np.pi / 180.0, 95.0 * np.pi / 180.0],
            "pitch_limits": [-0.0001, 0.0001],
            "dist_limits": [1000.0, 2000.0],
            "dist_terminal": 100.0,
        },
    }

    env = gym.make("flyer-v1", config=env_config)
    print(f"reset_obs: {env.reset()}")

    policy = SAC.load("models/flyer_heading_her_1000_2000-v1/best_model.zip", env=env)

    obs, info = env.reset()
    print(f"env: {env}")
    print(f"env.config: {env.config}")
    print(f"obs: {obs}")
    done = False

    observations = []
    times = []
    # targets = []
    dt = 1 / env_config["simulation_frequency"]
    time = 0.0

    while not done:
        action, _states = policy.predict(obs, deterministic=True)
        print(f"obs: {obs}")
        action = np.arctan2(obs["desired_goal"][1], obs["desired_goal"][0])
        print(f"action: {action * (180.0/np.pi)}")
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f'reward: {reward}')

        v_dict = env.unwrapped.vehicle.dict
        controls = env.unwrapped.vehicle.aircraft.controls
        # print(f'vehicle: {env.unwrapped.vehicle.aircraft.controls}')
        # print(f'v_dict: {v_dict}')

        obs_dict = {
            "elevator": controls["elevator"],
            "aileron": controls["aileron"],
            "rudder": 0.0,
            "x": v_dict["x"],
            "y": v_dict["y"],
            "z": v_dict["z"],
            "x_com": obs["desired_goal"][0],
            "y_com": obs["desired_goal"][1],
            "z_com": -1000.0,
            "pitch": v_dict["pitch"],
            "roll": v_dict["roll"],
            "yaw": v_dict["yaw"],
            "u": v_dict["u"],
            "reward": reward,
        }

        times.append(time)
        # targets.append(info['t_pos'])
        observations.append(obs_dict)
        time += dt

        if terminated or truncated:
            done = True

    env.close()
    observations = pd.DataFrame.from_dict(observations)
    plot_long(observations, times, env_config["duration"])
    plot_lat(observations, times, env_config["duration"])
    plot_track(observations)
    print(f'return: {np.sum(observations["reward"])}')
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
    fig.savefig("flyer_long_control_her.pdf")


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
    fig.savefig("flyer_lat_control_her.pdf")


def plot_track(outputs):

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.plot(outputs["x"], outputs["y"], c=COLOURS[1])
    ax.scatter(outputs["x_com"][1], outputs["y_com"][1], c=COLOURS[2])
    ax.set_ylabel(r"$y [m]$", fontsize=15)
    ax.set_xlabel(r"$x [m]$", fontsize=15)
    ax.set_aspect("equal")
    ax.grid()

    fig.savefig("flyer_track_control_her.pdf")


if __name__ == "__main__":
    main()
