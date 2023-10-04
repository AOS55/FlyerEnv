import numpy as np
import os
from typing import List, Tuple, Union, Optional
from flyer_env.utils import Vector
from simple_pid import PID
from pyflyer import Aircraft


class ControlledAircraft:
    """
    A wrapper around a pyflyer::Aircraft that allows for various preplanned controller actions
    """
    def __init__(self,
                 aircraft: Aircraft,
                 dt: float,
                 trim: Vector = [0.0, 0.04055471935347572, 0.6730648623679762, 0.0],
                 update_rate: dict = {
                     "low_level": 1/1000.0,
                     "mid_level": 1/100.0,
                     "high_level": 1/10.0
                 }
                ):
        
        self.dt = dt
        self.aircraft = aircraft
        self.update_rate = update_rate

        # path = os.path.join(*[os.path.dirname(os.path.realpath(__file__)), "..", "envs", "data/"])
        # self.aircraft = Aircraft(data_path=path)
        # self.aircraft.reset(position, heading, speed)

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

        self.low_time_since_pitch_update = 0.0
        self.low_time_since_roll_update = 0.0
        self.low_time_since_tla_update = 0.0
        self.mid_time_since_alt_update = 0.0
        self.mid_time_since_hdg_update = 0.0
        self.high_time_since_update = 0.0

        self.u_alt = 0.0
        self.u_hdg = 0.0
        self.hdg = 0.0

    def act(self, action: Union[dict, str] = None) -> None:
        
        self.low_time_since_pitch_update += self.dt
        self.low_time_since_roll_update += self.dt
        self.low_time_since_tla_update += self.dt
        self.mid_time_since_alt_update += self.dt
        self.mid_time_since_hdg_update += self.dt
        self.high_time_since_update += self.dt
        next_aileron, next_elevator, next_tla, next_rudder = self._trim
        aircraft_dict = self.aircraft.dict

        if action:
            
            # Low level controllers
            if "pitch" in action:
                com_pitch = action["pitch"]
                if self.low_time_since_pitch_update >= self.update_rate["low_level"]:
                    next_elevator = self.pitch_controller(aircraft_dict['pitch'] - com_pitch) + self._trim[1]
                    self.low_time_since_pitch_update = 0.0

            if "roll" in action:
                com_roll = action["roll"]
                if self.low_time_since_roll_update >= self.update_rate["low_level"]:
                    next_aileron = self.roll_controller(aircraft_dict['roll'] - com_roll)
                    self.low_time_since_roll_update = 0.0

            if "speed" in action:
                com_speed = action["speed"]
                if self.low_time_since_tla_update >= self.update_rate["low_level"]:
                    next_tla = self.speed_controller(aircraft_dict['u'] - com_speed)
                    self.low_time_since_tla_update = 0.0

            # High level controllers
            if "alt" in action:
                com_alt = action["alt"]
                next_elevator = self.alt_controller(aircraft_dict['z'] - com_alt, aircraft_dict['pitch'])

            if "heading" in action:
                com_hdg = self.clip_heading(action["heading"])
                next_aileron = self.heading_controller(com_hdg - aircraft_dict['yaw'], aircraft_dict['roll'])

            if "pursuit_target" in action:
                tgt_pos = action["pursuit_target"]
                next_aileron = self.pursuit_controller(tgt_pos,
                                                       np.array([aircraft_dict['x'], aircraft_dict['y']]),
                                                       aircraft_dict['yaw'],
                                                       aircraft_dict['roll'])
        
        action = {"aileron": next_aileron,
                  "elevator": next_elevator,
                  "tla": next_tla, 
                  "rudder": next_rudder}

        self.aircraft.act(action)
    
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
        if self.mid_time_since_alt_update >= self.update_rate["mid_level"]:
            alt_err = np.clip(alt_err, -200.0, 200.0)
            self.u_alt = self.pid_alt(-alt_err)
            self.mid_time_since_alt_update = 0.0
        pitch_err = pitch - self.u_alt
        elevator = self.pitch_controller(pitch_err)
        return elevator
    
    def heading_controller(self, hdg_err: float, bank: float) -> float:
        """
        PID based heading controller

        :param hdg_err: heading error [rad]
        :param bank: aircraft bank angle [rad]
        :return: aileron deflection [rad]
        """
        if self.mid_time_since_hdg_update >= self.update_rate["mid_level"]:
            hdg_err = np.clip(hdg_err, -10.0*(np.pi/180.0), 10.0*(np.pi/180.0))
            self.u_hdg = self.pid_heading(hdg_err)
            self.mid_time_since_hdg_update = 0.0
        bank_err = bank - self.u_hdg
        aileron = self.roll_controller(bank_err)
        return aileron
    
    def pursuit_controller(self, tgt_pos: Vector, ac_pos: Vector, ac_hdg: float, ac_bank: float) -> float:
        """
        Pure pursuit controller

        :param tgt_pos: (x, y) target position [m]
        :return: aileron deflection [-1, +1]
        """
        if self.high_time_since_update >= self.update_rate["high_level"]:
            pos_error = tgt_pos[0:2] - ac_pos
            self.hdg = self.clip_heading(np.arctan2(pos_error[1], pos_error[0]))
            self.high_time_since_update = 0.0
        hdg_err = self.hdg - ac_hdg
        aileron = self.heading_controller(hdg_err, ac_bank)
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

    def step(self, dt: float):
        """
        Use the step class found in PyAircraft
        """
        self.aircraft.step(dt)

    @property
    def dict(self):
        """
        Helper method to access dict from the underlying rust PyAircraft
        """
        return self.aircraft.dict

    @property
    def crashed(self):
        """
        Helper method to access crashed from the underlying rust PyAircraft
        """
        return self.aircraft.crashed
    
    @property
    def position(self):
        """
        Helper method to access position from the underlying rust PyAircraft
        """
        return self.aircraft.position
    
    @property
    def heading(self):
        """
        Helper method to access heading from the underlying rust PyAircraft
        """
        return self.aircraft.heading

    @staticmethod
    def clip_heading(heading: float) -> float:
        if -np.pi <= heading <= np.pi:
            pass
        elif -np.pi > heading:
            heading += 2.0 * np.pi
        elif np.pi < heading:
            heading -= 2.0 * np.pi
        return heading
