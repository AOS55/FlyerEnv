import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flyer_env.utils import Vector
from flyer_env.aircraft import TrackPoints

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


class PointBot:

    def __init__(self, start_pos: Vector, hdg: float, speed: float, dt: float):
        self.pos = start_pos
        self.hdg = hdg
        self.speed = (speed,)
        self.dt = dt

    def step(self, hdg: float, alt: float, speed: float):
        self.pos[-1] = alt
        self.pos[0] += np.cos(hdg) * speed * self.dt
        self.pos[1] += np.sin(hdg) * speed * self.dt


def test_trajectory():

    dt = 0.01
    # start_pos = np.array([0.0, 0.0])
    # target_points = [
    #     [0.0, 0.0],
    #     [24060.25390625, -15000.0],
    #     [24060.25390625, -5000.0],
    #     [15400.0, 0.0],
    #     [400.0, 0.0]
    # ]

    target_points = [
        [0.0, 0.0],
        [5000.0, -5000.0],
        [5000.0, -3000.0],
        [3000.0, 0.0],
        [400.0, 0.0],
    ]

    target_dicts = []
    for values in target_points:
        target_dicts.append({"x": values[0], "y": values[1]})
    targets = pd.DataFrame.from_dict(target_dicts)

    goal_set = {idx: target for (idx, target) in enumerate(target_points)}

    nav_track = TrackPoints(goal_set)
    speed = 80.0
    alt = -1000.0
    vehicle = PointBot(np.array([0.0, 0.0, -1000.0]), 0.0, speed, dt=dt)

    observations = []

    for idt in range(int(200 / dt)):

        hdg = nav_track.arc_path(vehicle.pos.copy())
        vehicle.step(hdg, alt, speed)
        obs_dict = {"x": vehicle.pos[0].copy(), "y": vehicle.pos[1].copy()}

        observations.append(obs_dict)

    observations = pd.DataFrame.from_dict(observations)
    plot_position(observations, targets)
    plt.show()


def plot_position(outputs, targets):

    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.plot(outputs["x"], outputs["y"], c=COLOURS[1])
    ax.scatter(targets["x"], targets["y"], c=COLOURS[2])
    ax.set_ylabel(r"$y [m]$", fontsize=15)
    ax.set_xlabel(r"$x [m]$", fontsize=15)
    ax.set_aspect("equal")
    ax.grid()

    fig.show()


if __name__ == "__main__":
    test_trajectory()
