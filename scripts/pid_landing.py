import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict
from flyer_env import utils
from flyer_env.aircraft import TrackPoints


plt.rcParams.update({
    "text.usetex": True
})

COLOURS = [[0, 18, 25], [0, 95, 115], [10, 147, 150], [148, 210, 189], [233, 216, 166],
               [238, 155, 0], [202, 103, 2], [187, 62, 3], [174, 32, 18], [155, 34, 38]]
COLOURS = [[value/255 for value in rgb] for rgb in COLOURS]


def main():

    env_config = {
        "duration": 600.0,
        "action": {
            "type": "ControlledAction"
        },
        "runway_configuration": {
            "runway_position": [0.0, 0.0],
            "runway_width": 20.0,
            "runway_length": 1500.0,
            "runway_heading": 180.0
        }
    }
    
    env = gym.make("runway-v1", config=env_config)
    obs, info = env.reset()
    terminated = truncated = False

    start_pos = env.unwrapped.vehicle.position
    # aircraft_pos = [-500.0, -500.0, -1000.0]
    touchdown_points = env.unwrapped.world.touchdown_points()

    side = np.sign((touchdown_points["faf"][0] - touchdown_points["touchdown"][0]) * (start_pos[1] - touchdown_points["faf"][1]) 
                   - (touchdown_points["touchdown"][1] - touchdown_points["faf"][1]) * (start_pos[0] - touchdown_points["faf"][0]))
    
    if side < 0:
        keys = ["iaf_l", "inaf_l", "faf", "touchdown"] 
        target_points = {x:touchdown_points[x] for x in keys}
    else:
        keys = ["iaf_r", "inaf_r", "faf", "touchdown"]
        target_points = {x:touchdown_points[x] for x in keys} 

    print(f'side: {side}, target_points: {target_points}')

    target_list = [v.copy() for v in target_points.values()]
    # [x.append(-1000.0) for x in target_list]
    target_list.insert(0, start_pos[0:2])
    
    target_dicts = []
    for values in touchdown_points.values():
        target_dicts.append({"x": values[0], "y": values[1]})
    targets = pd.DataFrame.from_dict(target_dicts)

    goal_set = {idx: target for (idx, target) in enumerate(target_list)}
    nav_track = TrackPoints(goal_set, radius=3000.0)

    speed = 100.0
    alt = -1000.0
    
    observations = []

    print(f'target_list: {target_list}')

    while not (terminated or truncated):

        pos = env.unwrapped.vehicle.position
        heading = nav_track.arc_path(pos)
        # print(f'heading_com: {heading * 180.0/np.pi}, heading_act: {env.unwrapped.vehicle.dict["yaw"] * 180.0/np.pi}')
        action = [np.sin(heading), np.cos(heading),
                  utils.lmap(alt, env.unwrapped.action_type.alt_range, [-1.0, 1.0]),
                  utils.lmap(speed, env.unwrapped.action_type.speed_range, [-1.0, 1.0])]
        obs, reward, terminated, truncated, info = env.step(action)


        v_dict = env.unwrapped.vehicle.dict
        controls = env.unwrapped.vehicle.aircraft.controls
        obs_dict = {
            'x': v_dict['x'],
            'y': v_dict['y']
        }
        observations.append(obs_dict)     
    env.close()

    observations = pd.DataFrame.from_dict(observations)
    plot_position(observations, targets)
    plt.show()

def plot_position(outputs, targets):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    ax.plot(outputs['x'], outputs['y'], c=COLOURS[1])
    ax.scatter(targets['x'], targets['y'], c=COLOURS[2])
    ax.set_ylabel(r"$y [m]$", fontsize=15)
    ax.set_xlabel(r"$x [m]$", fontsize=15)
    ax.set_aspect('equal')
    ax.grid()
    
    fig.show()

if __name__=="__main__":
    main()