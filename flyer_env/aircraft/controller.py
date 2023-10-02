import numpy as np
import os
from typing import List, Tuple, Union, Optional
from flyer_env.utils import Vector
from simple_pid import PID
from pyflyer import Aircraft


class ControlledAircraft:

    def __init__(self,
                 position: Vector = [0.0, 0.0, -1000.0],
                 heading: float = 0.0,
                 speed: float = 100.0,
                 trim: Vector = [0.0, 0.04055471935347572, 0.6730648623679762, 0.0]
                ):
        
        path = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "..", "envs", "data/"])
        self.aircraft = Aircraft(data_path=path)
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
        
        action = {"aileron": next_aileron,
                  "elevator": next_elevator,
                  "tla": next_tla, 
                  "rudder": next_rudder}

        self.aircraft.act(action)

        # return 
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
