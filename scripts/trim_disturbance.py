import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyflyer import Aircraft

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


def simulate():

    # Trim values for given airspeed and conditions
    pitch = -0.06081844622950141
    elevator = 0.04055471935347572
    tla = 0.6730648623679762

    aircraft = Aircraft()
    aircraft.reset([0.0, 0.0, -1000.0], 0.0, 100.0)
    dt = 0.01
    controls = []
    states = []
    times = []
    duration = 100.0

    for ids in range(int(duration / dt)):

        time = ids * dt

        # if 5.0 < time < 6.0:
        #     control_input = [0.0, -5.0 * np.pi/180.0, tla, 0.0]
        # elif 6.0 < time < 7.0:
        #     control_input = [0.0, 5.0 * np.pi/180.0, tla, 0.0]
        # if 12.0 < time < 13.0:
        #     control_input = [-5.0 * np.pi/180.0, elevator, tla, 0.0]
        # elif 13.0 < time < 14.0:
        #     control_input = [5.0 * np.pi/180.0, elevator, tla, 0.0]
        # elif 19.0 < time < 20.0:
        #     control_input = [0.0, elevator, tla, -5.0 * np.pi/180.0]
        # elif 20.0 < time < 21.0:
        #     control_input = [0.0, elevator, tla, 5.0 * np.pi/180.0]
        # else:
        #     control_input = [0.0, elevator, tla, 0.0]

        if 5.0 < time < 6.0:
            control_input = [-5.0 * np.pi / 180.0, elevator, tla, 0.0]
        elif 6.0 < time < 7.0:
            control_input = [5.0 * np.pi / 180.0, elevator, tla, 0.0]
        elif 20.0 < time < 21.0:
            control_input = [0.0, elevator, tla, -5.0 * np.pi / 180.0]
        elif 21.0 < time < 22.0:
            control_input = [0.0, elevator, tla, 5.0 * np.pi / 180.0]
        else:
            control_input = [0.0, elevator, tla, 0.0]

        control = {
            "aileron": control_input[0],
            "elevator": control_input[1],
            "tla": control_input[2],
            "rudder": control_input[3],
        }

        aircraft.act(control)
        aircraft.step(dt)
        controls.append(control)
        states.append(aircraft.dict)
        times.append(time)
    output = pd.DataFrame.from_dict(states)
    input = pd.DataFrame.from_dict(controls)
    plot_long(input, output, times)
    plot_lat(input, output, times)
    plt.show()


def plot_long(inputs, outputs, times):
    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Longitudinal Disturbance}", fontsize=30)
    ax[0].plot(times, inputs["elevator"] * (180.0 / np.pi), c=COLOURS[7])
    ax[0].set_ylabel(r"$\delta_{e} [^{\circ}]$", fontsize=15)

    ax[1].plot(times, outputs["q"] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].set_ylabel(r"$q [^{\circ}/s]$", fontsize=15)

    ax[2].plot(times, outputs["pitch"] * 180.0 / np.pi, c=COLOURS[1])
    ax[2].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[3].plot(times, outputs["u"], c=COLOURS[1])
    ax[3].set_ylabel(r"$V_{\infty} [m/s]$", fontsize=15)
    ax[3].set_xlabel(r"time [$s$]", fontsize=15)

    [axis.set_xlim(0.0, 100.0) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.savefig("long_trim_dist.pdf")


def plot_lat(inputs, outputs, times):
    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Lateral-Directional Disturbance}", fontsize=30)
    ax[0].plot(
        times, inputs["aileron"] * (180.0 / np.pi), c=COLOURS[5], label=r"aileron"
    )
    ax[0].plot(
        times,
        inputs["rudder"] * (180.0 / np.pi),
        c=COLOURS[7],
        linestyle="dashed",
        label=r"rudder",
    )
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r"\textbf{Control}")

    ax[1].plot(times, outputs["p"] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].set_ylabel(r"$p [^{\circ}/s]$", fontsize=15)

    ax[2].plot(times, outputs["r"] * 180.0 / np.pi, c=COLOURS[1])
    ax[2].set_ylabel(r"$r [^{\circ}/s]$", fontsize=15)

    ax[3].plot(times, outputs["roll"] * 180.0 / np.pi, c=COLOURS[1], label=r"$\phi$")
    ax[3].plot(
        times,
        outputs["yaw"] * 180.0 / np.pi,
        c=COLOURS[2],
        linestyle="dashed",
        label=r"$\psi$",
    )
    ax[3].set_ylabel(r"attitude $[^{\circ}]$", fontsize=15)
    ax[3].set_xlabel(r"time [$s$]", fontsize=15)
    ax[3].legend(title=r"\textbf{Attitude}")

    [axis.set_xlim(0.0, 40.0) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.savefig("lat_trim_dist.pdf")


def main():
    simulate()


if __name__ == "__main__":
    main()
