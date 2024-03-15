import pytest

from pyflyer import Aircraft

aircraft_types = ["TO"]
trim_results = {
    "TO": [0.0028701873057483977, 0.04046632028946647, 0.763808910861435]
}


@pytest.mark.parametrize("aircraft_type", aircraft_types)
def test_trim(aircraft_type):
    aircraft = Aircraft(aircraft_type)
    altitude = 1000.0  # trim altitude to maintain
    airspeed = 100.0  # trim airspeed to maintain
    n_steps = 100  # Number of PSO trim iterations
    trim = aircraft.trim(altitude, airspeed, n_steps)
    assert trim[0] == pytest.approx(0)  # aileron trim position
    assert trim[1] == pytest.approx(trim_results[aircraft_type][1])  # elevator trim position
    assert trim[2] == pytest.approx(trim_results[aircraft_type][2])  # TLA trim position
