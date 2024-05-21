import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from flyer_env.utils import lmap 
from stable_baselines3 import SAC

plt.rcParams.update({
    "text.usetex": True
})

COLOURS = [[0, 18, 25], [0, 95, 115], [10, 147, 150], [148, 210, 189], [233, 216, 166],
               [238, 155, 0], [202, 103, 2], [187, 62, 3], [174, 32, 18], [155, 34, 38]]
COLOURS = [[value/255 for value in rgb] for rgb in COLOURS]


def main():

    env_config = {
        "observation": {
            "type": "Trajectory"
        },
        "action": {
            "type": "ContinuousAction"
        },
        "area": (256, 256),
        "simulation_frequency": 100.0,
        "duration": 10.0,
        "trajectory_config": {
            "name": "sl",
            "length": 10.0
        }
    }

    env = gym.make("trajectory-v1", config=env_config)
    policy = SAC.load('models/trajectory_full-v1/best_model.zip')

    control_env_config = {
        "observation": {
            "type": "Dynamics"
        },
        "action": {
            "type": "ControlledAction"
        },
        "area": (256, 256),
        "simulation_frequency": 100.0,
        "duration": 10.0
    }

    c_env = gym.make("control-v1", config=control_env_config)
    c_obs, c_info = c_env.reset()
    c_done = False
    c_action = [0.0, 1.0,
        lmap(-1000.0, [0.0, -10000], [-1.0, 1.0]),
        lmap(100.0, [60.0, 110.0], [-1.0, 1.0])]
    
    obs, info = env.reset()
    done = False

    observations = []
    times = []
    dt = 1/env_config["simulation_frequency"]
    time = 0.0

    while not done:
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        v_dict = env.unwrapped.vehicle.dict
        controls = env.unwrapped.vehicle.controls

        obs_dict = {
            'elevator': controls['elevator'],
            'aileron': controls['aileron'],
            'tla': controls['tla'],
            'rudder': controls['rudder'],
            'x': v_dict['x'],
            'y': v_dict['y'],
            'z': v_dict['z'],
            'pitch': v_dict['pitch'],
            'roll': v_dict['roll'],
            'yaw': v_dict['yaw'],
            'u': v_dict['u'],
            'reward': reward
        }

        c_obs, c_reward, c_terminated, c_truncated, c_info = c_env.step(c_action)

        c_v_dict = c_env.unwrapped.vehicle.dict
        c_controls = c_env.unwrapped.vehicle.controls

        c_obs_dict = {
            'c_elevator': c_controls[1],
            'c_aileron': c_controls[0],
            'c_tla': c_controls[2],
            'c_rudder': c_controls[3],
            'c_x': c_v_dict['x'],
            'c_y': c_v_dict['y'],
            'c_z': c_v_dict['z'],
            'c_pitch': c_v_dict['pitch'],
            'c_roll': c_v_dict['roll'],
            'c_yaw': c_v_dict['yaw'],
            'c_u': c_v_dict['u'],
            'c_reward': c_reward, 
        }
        obs_dict.update(c_obs_dict)

        times.append(time)
        observations.append(obs_dict)
        # observations.append(c_obs_dict)
        time += dt

        if terminated or truncated:
            done = True

    env.close()
    observations = pd.DataFrame.from_dict(observations)
    plot_long(observations, times, env_config["duration"])
    plot_lat(observations, times, env_config["duration"])
    # plot_track(observations)
    # plt.show()


def plot_long(outputs, times, exp_len):
    fig, ax = plt.subplots(5, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Longitudinal Tracking}")
    ax[0].plot(times, outputs['elevator'], c=COLOURS[5], label=r'elevator')
    ax[0].plot(times, outputs['c_elevator'], linestyle='dashed' , c=COLOURS[5])
    ax[0].plot(times, outputs['tla'], c=COLOURS[6], label=r'tla')
    ax[0].plot(times, outputs['tla'], linestyle='dashed', c=COLOURS[6])
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r'\textbf{Control}')

    ax[1].plot(times, outputs['pitch'] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].plot(times, outputs['c_pitch'] * 180.0 / np.pi, linestyle='dashed', c=COLOURS[1])
    ax[1].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[2].plot(times, outputs['z'], c=COLOURS[1])
    ax[2].plot(times, outputs['c_z'], linestyle='dashed', c=COLOURS[1])
    # ax[2].plot(times, outputs['z_com'], linestyle='dashed', c=COLOURS[2])
    ax[2].set_ylabel(r"$z [m]$", fontsize=15)

    ax[3].plot(times, outputs['u'], c=COLOURS[1])
    ax[3].plot(times, outputs['c_u'], linestyle='dashed', c=COLOURS[1])
    ax[3].set_ylabel(r"$u [\frac{m}{s}]$", fontsize=15)

    ax[4].plot(times, outputs['reward'], c=COLOURS[1])
    ax[4].plot(times, outputs['c_reward'], linestyle='dashed', c=COLOURS[1])
    ax[4].set_ylabel(r'Reward', fontsize=15)
    ax[4].set_xlabel(r'time [$s$]', fontsize=15)

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.savefig("traj_long.pdf")

def plot_lat(outputs, times, exp_len):

    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(10)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Lateral-Directional Tracking}")
    ax[0].plot(times, outputs['aileron'], c=COLOURS[5], label=r'aileron')
    ax[0].plot(times, outputs['rudder'], c=COLOURS[7], linestyle='dashed', label=r'rudder')
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r'\textbf{Control}')

    ax[1].plot(times, outputs['roll'] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].plot(times, outputs['yaw'] * 180.0 / np.pi, c=COLOURS[2], linestyle='dashed', label=r'$\psi$')
    ax[1].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[2].plot(times, outputs['u'], c=COLOURS[1])
    ax[2].set_ylabel(r"$u [\frac{m}{s}]$", fontsize=15)

    ax[3].plot(times, outputs['reward'], c=COLOURS[1])
    ax[3].set_ylabel(r'Reward', fontsize=15)
    ax[3].set_xlabel(r'time [$s$]', fontsize=15)


    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.savefig("traj_lat.pdf")
    fig.show()

def plot_track(outputs):

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.plot(outputs['x'], outputs['y'], c=COLOURS[1])
    # ax.plot(outputs['x_com'], outputs['y_com'], linestyle='dashed', c=COLOURS[2])
    ax.set_ylabel(r"$y [m]$", fontsize=15)
    ax.set_xlabel(r"$x [m]$", fontsize=15)
    ax.set_aspect('equal')
    ax.grid()

    fig.show()

if __name__=="__main__":
    main()
