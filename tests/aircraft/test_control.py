import os
import numpy as np
import pytest
from flyer_env.aircraft.controller import ControlledAircraft
from pyflyer import Aircraft

FPS = 1000
init_conditions = {"pos": [0.0, 0.0, -1000.0], "heading": 0.0, "airspeed": 100.0}
step_results = {
    'x': 0.999735471749434, 'y': 0.0, 'z': -1000.0000602366912, 
    'pitch': 3.3100002208511556e-05, 'yaw': 0.0, 'roll': 0.0,
    'u': 99.94710404280157, 'v': 0.0, 'w': -0.01105721758462818,
    'p': 0.0, 'q': 0.005843277236213746, 'r': 0.0
}
# Only tests for TO aircraft type as the only gains provided are for this aircraft


def test_step():
    a = _get_controlled_aircraft()
    for _ in range(10):
        a.step(dt = 1 / FPS)
    a_dict = a.dict
    assert step_results["x"] == pytest.approx(a_dict["x"])
    assert step_results["y"] == pytest.approx(a_dict["y"])
    assert step_results["z"] == pytest.approx(a_dict["z"])
    assert step_results["pitch"] == pytest.approx(a_dict["pitch"])
    assert step_results["roll"] == pytest.approx(a_dict["roll"])
    assert step_results["yaw"] == pytest.approx(a_dict["yaw"])
    assert step_results["u"] == pytest.approx(a_dict["u"])
    assert step_results["v"] == pytest.approx(a_dict["v"])
    assert step_results["w"] == pytest.approx(a_dict["w"])
    assert step_results["p"] == pytest.approx(a_dict["p"])
    assert step_results["q"] == pytest.approx(a_dict["q"])
    assert step_results["r"] == pytest.approx(a_dict["r"])


def test_pitch():
    pitch_com = -0.5 * np.pi/180.0
    max_steps = 1e6

    action = {"pitch": pitch_com}
    a = _get_controlled_aircraft()
    ids = 0
    while a.dict["pitch"] != pytest.approx(pitch_com, rel=1e-2):
        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            print("Reached max steps")
            assert False
    assert True


def test_roll():

    roll_com = 0.1 * np.pi/180.0
    max_steps = 1e6

    action = {"roll": roll_com}
    a = _get_controlled_aircraft()
    ids = 0
    while a.dict["roll"] != pytest.approx(roll_com, rel=1e-2):
        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            assert False
    assert True


def test_speed():

    speed_com = 110.0
    max_steps = 1e6

    action = {"speed": speed_com}
    a = _get_controlled_aircraft()
    ids = 0
    while a.dict["u"] != pytest.approx(speed_com, rel=1e-2):
        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            assert False
    assert True


def test_alt():

    alt_com = -2000.0
    max_steps = 1e6

    action = {"alt": alt_com}
    a = _get_controlled_aircraft()
    ids = 0
    while a.dict["z"] != pytest.approx(alt_com, rel=10.0):
        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            assert False
    assert True


def test_heading():

    hdg_com = 10.0 * np.pi/180.0
    max_steps = 1e6

    action = {"heading": hdg_com}
    print(f'action: {action}')
    a = _get_controlled_aircraft()
    a.act(action)
    ids = 0
    while a.dict["yaw"] != pytest.approx(hdg_com, rel=1e-2):
        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            assert False
    assert True


def test_pursuit_target():

    tgt_com = np.array([10000.0, 10000.0])
    max_steps = 1e6

    action = {"pursuit_target": tgt_com}
    a = _get_controlled_aircraft()
    ids = 0
        
    while 0.0 != pytest.approx(np.linalg.norm(tgt_com-np.array([a.dict['x'], a.dict['y']])), rel=1e2):

        a.act(action)
        a.step(dt = 1 / FPS)
        ids += 1
        if ids > max_steps:
            assert False
    assert True

# TODO: Add a method to test track points


def _get_controlled_aircraft():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    acft = Aircraft(aircraft_name="TO", data_path = path)
    acft.reset(pos=init_conditions["pos"], heading=init_conditions["heading"], airspeed=init_conditions["airspeed"])
    return ControlledAircraft(aircraft=acft, dt=1/FPS)