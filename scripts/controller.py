import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Union, Optional
from flyer_env.utils import Vector
from simple_pid import PID
from pyflyer import Aircraft

plt.rcParams.update({
    "text.usetex": True
})

COLOURS = [[0, 18, 25], [0, 95, 115], [10, 147, 150], [148, 210, 189], [233, 216, 166],
               [238, 155, 0], [202, 103, 2], [187, 62, 3], [174, 32, 18], [155, 34, 38]]
COLOURS = [[value/255 for value in rgb] for rgb in COLOURS]

class ControlledAircraft:

    def __init__(self,
                 position: Vector = [0.0, 0.0, -1000.0],
                 heading: float = 0.0,
                 speed: float = 100.0,
                 trim: Vector = [0.0, 0.04055471935347572, 0.6730648623679762, 0.0]
                ):
        
        self.aircraft = Aircraft()
        self.aircraft.reset(position, heading, speed)

        self.pid_pitch = PID(10.0, 2.0, 0.1)
        self.pid_pitch.output_limits = (-5.0 * (np.pi/180.0), 30.0 * (np.pi/180.0))

        self.pid_roll = PID(1.0, 0.1, 0.01)
        self.pid_roll.output_limits = (-5.0 * (np.pi/180.0), 5.0 * (np.pi/180.0))

        self.pid_speed = PID(0.2, 1.0, 0.0, setpoint=trim[2])

        self.pid_alt = PID(0.000873, 0.000, 0.0)
        self.pid_alt.output_limits = (-20.0, 20.0)

        self.pid_heading = PID(-3.0, 0.0, 0.0)

        self._trim = trim
        self.controls = trim
        self.limits = [6.0*(np.pi/180.0), 6.0*(np.pi/180.0), 0.2, 6.0*(np.pi/180.0)]

    def act(self, action: Union[dict, str] = None) -> None:

        next_aileron, next_elevator, next_tla, next_rudder = self._trim
        aircraft_dict = self.aircraft.dict

        if action:
            
            # Low level controllers
            if "pitch" in action:
                com_pitch = action["pitch"]
                next_elevator = self.pitch_controller(aircraft_dict['pitch'] - com_pitch) + self._trim[1]

            if "roll" in action:
                com_roll = action["roll"]
                next_aileron = self.roll_controller(aircraft_dict['roll'] - com_roll)

            if "speed" in action:
                com_speed = action["speed"]
                next_tla = self.speed_controller(aircraft_dict['u'] - com_speed)

            # High level controllers
            if "alt" in action:
                com_alt = action["alt"]
                next_elevator = self.alt_controller(aircraft_dict['z'] - com_alt, aircraft_dict['pitch'])

            if "heading" in action:
                com_hdg = self.clip_heading(action["heading"])
                next_aileron = self.heading_controller(com_hdg - aircraft_dict['yaw'], aircraft_dict['roll'])

            if "pursuit_target" in action:
                tgt_pos = action["pursuit_target"]
                next_aileron = self.pursuit_controller(tgt_pos)
        
        return [next_aileron, next_elevator, next_tla, next_rudder]
        # return self.rate_limit(next_controls)
    
    def pitch_controller(self, pitch_err: float) -> float:
        """
        PID based pitch controller

        :param pitch_err: pitch error [rad]
        :return: elevator deflection [rad]
        """
        elevator = self.pid_pitch(-pitch_err)
        return np.clip(elevator, -5.0 * (np.pi/180.0), 30.0 * (np.pi/180.0))

    def roll_controller(self, roll_err: float) -> float:
        """
        PID based roll controller

        :param roll_err: roll error [rad]
        :return: aileron deflection [rad]
        """
        aileron = self.pid_roll(-roll_err)
        return np.clip(aileron, -10.0 * (np.pi/180.0), 10.0 * (np.pi/180.0))

    def speed_controller(self, speed_err: float) -> float:
        """
        PID based speed controller

        :param speed_err: speed error [m/s]
        :return: tla deflection [rad]
        """
        tla = self.pid_speed(np.clip(speed_err, -5.0, 5.0))
        return np.clip(tla, 0.0, 1.0)
        
    def alt_controller(self, alt_err: float, pitch: float) -> float:
        """
        PID based altitude controller

        :param alt_err: altitude error [m]
        :param pitch: aircraft pitch attitude [rad]
        :return: elevator deflection [rad]
        """
        alt_err = np.clip(alt_err, -200.0, 200.0)
        u_alt = self.pid_alt(-alt_err)
        pitch_err = pitch - u_alt
        elevator = self.pitch_controller(pitch_err)
        return elevator
    
    def heading_controller(self, hdg_err: float, bank: float) -> float:
        """
        PID based heading controller

        :param hdg_err: heading error [rad]
        :param bank: aircraft bank angle [rad]
        :return: aileron deflection [rad]
        """
        hdg_err = np.clip(hdg_err, -10.0*(np.pi/180.0), 10.0*(np.pi/180.0))
        u_hdg = self.pid_heading(hdg_err)
        bank_err = bank - u_hdg
        aileron = self.roll_controller(bank_err)
        return aileron
    
    def pursuit_controller(self, tgt_pos: Vector) -> float:
        """
        Pure pursuit controller

        :param tgt_pos: (x, y) target position [m]
        :return: aileron deflection [-1, +1]
        """
        pos_error = tgt_pos[0:2] - self.position[0:2]
        heading = self.clip_heading(np.arctan2(pos_error[1], pos_error[0]))
        hdg_err = heading - self.heading
        aileron = self.heading_controller(hdg_err, self.bank)
        return aileron

    # # TODO: Is there a more elegant way of solving this, it's a little slow
    # def rate_limit(self, next_controls):
    #     controls = []
    #     for idc, limit in zip(range(4), self.limits):
    #         delta = next_controls[idc] - self.controls[idc]
    #         delta = np.clip(delta, -limit*self.dt, limit*self.dt)
    #         controls.append(self.controls[idc] + delta)
    #     self.controls = controls
    #     return controls

    @staticmethod
    def clip_heading(heading: float) -> float:
        if -np.pi <= heading <= np.pi:
            pass
        elif -np.pi > heading:
            heading += 2.0 * np.pi
        elif np.pi < heading:
            heading -= 2.0 * np.pi
        return heading
        

def simulate():

    # Trim values for given airspeed and conditions
    pitch = -0.06081844622950141
    elevator = 0.04055471935347572
    tla = 0.6730648623679762

    dt = 0.001
    c_ac = ControlledAircraft()
    controls = []
    states = []
    times = []

    exp_len = 100.0

    action = {"speed": 100.0, "alt": -1500.0, "heading": 50.0*(np.pi/180.0)}
    # action = {"speed": 100.0, "roll": 0.0*(np.pi/180.0), "pitch": 10.0 * np.pi/180.0}
    for ids in range(int(exp_len/dt)):
        time = ids*dt
        # print(f"z: {c_ac.aircraft.dict['z']}")
        control_input = c_ac.act(action)
        c_ac.aircraft.step(dt, control_input)
        controls.append({"aileron": control_input[0],
                         "elevator": control_input[1],
                         "tla": control_input[2],
                         "rudder": control_input[3]})
        states.append(c_ac.aircraft.dict)
        times.append(time)

    output = pd.DataFrame.from_dict(states)
    input = pd.DataFrame.from_dict(controls)

    plot_long(input, output, times, exp_len)
    plot_lat(input, output, times, exp_len)
    plot_alt(input, output, times, exp_len)
    plt.show()


def plot_long(inputs, outputs, times, exp_len):
    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Longitudinal Disturbance}", fontsize=30)
    ax[0].plot(times, inputs['elevator'], c=COLOURS[5], label=r'elevator')
    # ax[0].plot(times, inputs['tla'], c=COLOURS[7], linestyle='dashed', label=r'tla')
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r'\textbf{Control}')

    ax[1].plot(times, outputs['q'] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].set_ylabel(r"$q [^{\circ}/s]$", fontsize=15)

    ax[2].plot(times, outputs['pitch'] * 180.0 / np.pi, c=COLOURS[1])
    ax[2].set_ylabel(r"$\theta [^{\circ}]$", fontsize=15)

    ax[3].plot(times, outputs['u'], c=COLOURS[1])
    ax[3].set_ylabel(r"$V_{\infty} [m/s]$", fontsize=15)
    ax[3].set_xlabel(r"time [$s$]", fontsize=15)

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.show()

def plot_lat(inputs, outputs, times, exp_len):
    fig, ax = plt.subplots(4, 1, sharex=True)
    [axis.grid() for axis in ax]
    fig.subplots_adjust(hspace=0.0)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Lateral-Directional Disturbance}", fontsize=30)
    ax[0].plot(times, inputs['aileron'], c=COLOURS[5], label=r'aileron')
    ax[0].plot(times, inputs['rudder'], c=COLOURS[7], linestyle='dashed', label=r'rudder')
    ax[0].set_ylabel(r"$\delta [^{\circ}]$", fontsize=15)
    ax[0].legend(title=r'\textbf{Control}')

    ax[1].plot(times, outputs['p'] * 180.0 / np.pi, c=COLOURS[1])
    ax[1].set_ylabel(r"$p [^{\circ}/s]$", fontsize=15)

    ax[2].plot(times, outputs['r'] * 180.0 / np.pi, c=COLOURS[1])
    ax[2].set_ylabel(r"$r [^{\circ}/s]$", fontsize=15)

    ax[3].plot(times, outputs['roll'] * 180.0 / np.pi, c=COLOURS[1], label=r'$\phi$')
    ax[3].plot(times, outputs['yaw'] * 180.0 / np.pi, c=COLOURS[2], linestyle='dashed', label=r'$\psi$')
    ax[3].set_ylabel(r"attitude $[^{\circ}]$", fontsize=15)
    ax[3].set_xlabel(r"time [$s$]", fontsize=15)
    ax[3].legend(title=r'\textbf{Attitude}')

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax]
    fig.show()

def plot_alt(inputs, outputs, times, exp_len):
    
    fig, ax = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(20)

    ax[0].set_title(r"\textbf{Altitude Disturbance}", fontsize=30)
    ax[0].plot(times, outputs['z'], c=COLOURS[1])
    ax[0].set_ylabel(r"$z [m]$", fontsize=15)

    [axis.set_xlim(0.0, exp_len) for axis in ax]
    [axis.xaxis.set_tick_params(labelsize=15) for axis in ax]
    [axis.yaxis.set_tick_params(labelsize=15) for axis in ax] 
    fig.show()

def main():
    simulate()


if __name__=="__main__":
    main()