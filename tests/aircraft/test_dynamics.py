import pytest
import os

from pyflyer import Aircraft

FPS = 100
aircraft_types = ["TO"]
init_conditions = {
    "TO": {"pos": [0.0, 0.0, -1000.0], "heading": 0.0, "airspeed": 100.0} 
}
step_results = {
    "TO": {
        'x': 0.9997353856120385, 'y': 0.0, 'z': -1000.0000364373016, 
        'pitch': 4.103064655067453e-05, 'yaw': 0.0, 'roll': 0.0,
        'u': 99.94707660278598, 'v': 0.0, 'w': -0.0031862052403734326,
        'p': 0.0, 'q': 0.008206129310134905, 'r': 0.0}
}
elevator_results = {
    "TO": {
        'x': 0.999735675675089, 'y': 0.0, 'z': -1000.0007494680681,
        'pitch': -0.0011294225340001502, 'yaw': 0.0, 'roll': 0.0,
        'u': 99.94693813079581, 'v': 0.0, 'w': -0.2627860427356989,
        'p': 0.0, 'q': -0.22588450680002922, 'r': 0.0
    }
}
aileron_results = {
    "TO": {
        'x': 0.9997353896978778, 'y': -0.00012511021181731513, 'z': -1000.0000363903944,
        'pitch': 4.1181102625689824e-05, 'roll': -0.001724717875755119, 'yaw': 0.0001501950743377316,
        'u': 99.94707296469151, 'v': -0.040060292615306595, 'w': -0.0032258822841918187,
        'p': -0.3449441238329653, 'q': 0.008214498845871889, 'r': 0.030045691550989573
    }
}
tla_results = {
    "TO": {
        'x': 1.0000800151352767, 'y': 0.0, 'z': -1000.0000364396583,
        'pitch': 4.103064655067453e-05, 'roll': 0.0, 'yaw': 0.0,
        'u': 100.01600250740947, 'v': 0.0, 'w': -0.003184319857419175,
        'p': 0.0, 'q': 0.008206129310134905, 'r': 0.0
    }
}


@pytest.mark.parametrize("aircraft_type", aircraft_types)
def test_step(aircraft_type):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    a = Aircraft(aircraft_name=aircraft_type, data_path = path)
    init = init_conditions[aircraft_type]
    a.reset(pos = init["pos"], heading=init["heading"], airspeed=init["airspeed"])
    a.step(dt = 1 / FPS)
    a_dict = a.dict
    assert step_results[aircraft_type]["x"] == pytest.approx(a_dict["x"])
    assert step_results[aircraft_type]["y"] == pytest.approx(a_dict["y"])
    assert step_results[aircraft_type]["z"] == pytest.approx(a_dict["z"])
    assert step_results[aircraft_type]["pitch"] == pytest.approx(a_dict["pitch"])
    assert step_results[aircraft_type]["roll"] == pytest.approx(a_dict["roll"])
    assert step_results[aircraft_type]["yaw"] == pytest.approx(a_dict["yaw"])
    assert step_results[aircraft_type]["u"] == pytest.approx(a_dict["u"])
    assert step_results[aircraft_type]["v"] == pytest.approx(a_dict["v"])
    assert step_results[aircraft_type]["w"] == pytest.approx(a_dict["w"])
    assert step_results[aircraft_type]["p"] == pytest.approx(a_dict["p"])
    assert step_results[aircraft_type]["q"] == pytest.approx(a_dict["q"])
    assert step_results[aircraft_type]["r"] == pytest.approx(a_dict["r"])


@pytest.mark.parametrize("aircraft_type", aircraft_types)
def test_elevator(aircraft_type):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    a = Aircraft(aircraft_name=aircraft_type, data_path = path)
    init = init_conditions[aircraft_type]
    a.reset(pos = init["pos"], heading=init["heading"], airspeed=init["airspeed"])
    action = {"elevator": 1.0, "aileron": 0.0, "tla": 0.0, "rudder": 0.0}
    a.act(action)
    a.step(dt = 1 / FPS)
    a_dict = a.dict
    assert elevator_results[aircraft_type]["x"] == pytest.approx(a_dict["x"])
    assert elevator_results[aircraft_type]["y"] == pytest.approx(a_dict["y"])
    assert elevator_results[aircraft_type]["z"] == pytest.approx(a_dict["z"])
    assert elevator_results[aircraft_type]["pitch"] == pytest.approx(a_dict["pitch"])
    assert elevator_results[aircraft_type]["roll"] == pytest.approx(a_dict["roll"])
    assert elevator_results[aircraft_type]["yaw"] == pytest.approx(a_dict["yaw"])
    assert elevator_results[aircraft_type]["u"] == pytest.approx(a_dict["u"])
    assert elevator_results[aircraft_type]["v"] == pytest.approx(a_dict["v"])
    assert elevator_results[aircraft_type]["w"] == pytest.approx(a_dict["w"])
    assert elevator_results[aircraft_type]["p"] == pytest.approx(a_dict["p"])
    assert elevator_results[aircraft_type]["q"] == pytest.approx(a_dict["q"])
    assert elevator_results[aircraft_type]["r"] == pytest.approx(a_dict["r"])


@pytest.mark.parametrize("aircraft_type", aircraft_types)
def test_aileron(aircraft_type):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    a = Aircraft(aircraft_name=aircraft_type, data_path = path)
    init = init_conditions[aircraft_type]
    a.reset(pos = init["pos"], heading=init["heading"], airspeed=init["airspeed"])
    action = {"elevator": 0.0, "aileron": 1.0, "tla": 0.0, "rudder": 0.0}
    a.act(action)
    a.step(dt = 1 / FPS)
    a_dict = a.dict
    assert aileron_results[aircraft_type]["x"] == pytest.approx(a_dict["x"])
    assert aileron_results[aircraft_type]["y"] == pytest.approx(a_dict["y"])
    assert aileron_results[aircraft_type]["z"] == pytest.approx(a_dict["z"])
    assert aileron_results[aircraft_type]["pitch"] == pytest.approx(a_dict["pitch"])
    assert aileron_results[aircraft_type]["roll"] == pytest.approx(a_dict["roll"])
    assert aileron_results[aircraft_type]["yaw"] == pytest.approx(a_dict["yaw"])
    assert aileron_results[aircraft_type]["u"] == pytest.approx(a_dict["u"])
    assert aileron_results[aircraft_type]["v"] == pytest.approx(a_dict["v"])
    assert aileron_results[aircraft_type]["w"] == pytest.approx(a_dict["w"])
    assert aileron_results[aircraft_type]["p"] == pytest.approx(a_dict["p"])
    assert aileron_results[aircraft_type]["q"] == pytest.approx(a_dict["q"])
    assert aileron_results[aircraft_type]["r"] == pytest.approx(a_dict["r"]) 


@pytest.mark.parametrize("aircraft_type", aircraft_types)
def test_tla(aircraft_type):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    a = Aircraft(aircraft_name=aircraft_type, data_path = path)
    init = init_conditions[aircraft_type]
    a.reset(pos = init["pos"], heading=init["heading"], airspeed=init["airspeed"])
    action = {"elevator": 0.0, "aileron": 0.0, "tla": 1.0, "rudder": 0.0}
    a.act(action)
    a.step(dt = 1 / FPS)
    a_dict = a.dict
    assert tla_results[aircraft_type]["x"] == pytest.approx(a_dict["x"])
    assert tla_results[aircraft_type]["y"] == pytest.approx(a_dict["y"])
    assert tla_results[aircraft_type]["z"] == pytest.approx(a_dict["z"])
    assert tla_results[aircraft_type]["pitch"] == pytest.approx(a_dict["pitch"])
    assert tla_results[aircraft_type]["roll"] == pytest.approx(a_dict["roll"])
    assert tla_results[aircraft_type]["yaw"] == pytest.approx(a_dict["yaw"])
    assert tla_results[aircraft_type]["u"] == pytest.approx(a_dict["u"])
    assert tla_results[aircraft_type]["v"] == pytest.approx(a_dict["v"])
    assert tla_results[aircraft_type]["w"] == pytest.approx(a_dict["w"])
    assert tla_results[aircraft_type]["p"] == pytest.approx(a_dict["p"])
    assert tla_results[aircraft_type]["q"] == pytest.approx(a_dict["q"])
    assert tla_results[aircraft_type]["r"] == pytest.approx(a_dict["r"])

    