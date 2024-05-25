import pytest
import os

from pyflyer import Aircraft

aircraft_types = ["TO"]  # Only setup for TO for now
trim_results = {
    "TO": [0.0028701873057483977, 0.04046632028946647, 0.763808910861435]
}

# TODO: Test trim for now
# @pytest.mark.parametrize("aircraft_type", aircraft_types)
# def test_trim(aircraft_type):
#     # TODO: Look at way to make trim routine more robust
#     path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
#     aircraft = Aircraft(data_path = path)
#     altitude = 1000.0  # trim altitude to maintain
#     airspeed = 100.0  # trim airspeed to maintain
#     n_steps = 100  # Number of PSO trim iterations
#     trim = aircraft.trim(altitude, airspeed, n_steps)
#     assert trim[0] == pytest.approx(trim_results[aircraft_type][0], abs=1e-1)  # pitch attitude
#     assert trim[1] == pytest.approx(trim_results[aircraft_type][1], rel=1e-2)  # elevator trim position
#     assert trim[2] == pytest.approx(trim_results[aircraft_type][2], rel=1e-1)  # TLA trim position
