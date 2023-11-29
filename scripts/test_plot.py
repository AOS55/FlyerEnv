import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC

plt.rcParams.update({
    "text.usetex": True
})

COLOURS = [[0, 18, 25], [0, 95, 115], [10, 147, 150], [148, 210, 189], [233, 216, 166], [238, 155, 0], [202, 103, 2], [187, 62, 3], [174, 32, 18], [155, 34, 38]]
COLOURS = [[value/255 for value in rgb] for rgb in COLOURS]

def main():

    env = gym.make("Pendulum-v1")
    policy = SAC.load("models/sac_pendulum-v1/best_model.zip")

    obs, info = env.reset()
    done = False
    observations = []
    dt = 0.05
    times = []
    time = 0.0

    while not done:
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        obs_dict = {
            'x': obs[0],
            'y': obs[1],
            'theta_dot': obs[2],
            'action': action,
            'reward': reward
        }
        times.append(time)
        observations.append(obs_dict)
        time += dt

        if terminated or truncated:
            done = True
    env.close()

    observations = pd.DataFrame.from_dict(observations)
    plot_pendulum(observations, times)

def plot_pendulum(outputs, times):
    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Pendulum Swingup}")
    ax[0].plot(times, outputs['x'], c=COLOURS[1], label=r'x')
    ax[0].set_ylabel(r'$x [m]$')

    ax[1].plot(times, outputs['y'], c=COLOURS[1], label=r'y')
    ax[1].set_ylabel(r'$y [m]$')

    ax[2].plot(times, outputs['action'], c=COLOURS[1], label=r'tau')
    ax[2].set_ylabel(r'$\tau [Nm]$')

    ax[3].plot(times, outputs['reward'], c=COLOURS[1])
    ax[3].set_ylabel(r'$reward [-]$')

    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.savefig("test_pendulum.pdf")

if __name__=="__main__":
    main()
