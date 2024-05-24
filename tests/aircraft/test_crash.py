from pyflyer import Aircraft
import os
import numpy as np
import pytest

crash_type = [
    "long-overspeed",
    "lat-overspeed",
    "over-rotate",
    "under-rotate",
    "high"
]

spec = {
    "long-overspeed": {"position": [0.0, 0.0, -2.0], "velocity": [300.0, 0.0, 0.0], "attitude": [0.0, 0.0, 0.0], "crashed": True},
    "lat-overspeed": {"position": [0.0, 0.0, -2.0], "velocity": [0.0, 0.0, 100.0], "attitude": [0.0, 0.0, 0.0], "crashed": True},
    "over-rotate": {"position": [0.0, 0.0, -2.0], "velocity": [0.0, 0.0, 0.0], "attitude": [40.0*np.pi/180.0, 0.0, 0.0], "crashed": True},
    "under-rotate": {"position": [0.0, 0.0, -2.0], "velocity": [0.0, 0.0, 0.0], "attitude": [-10.0*np.pi/180.0, 0.0, 0.0], "crashed": True},
    "high": {"position": [0.0, 0.0, -1000.0], "velocity": [500.0, 0.0, 30.0], "attitude": [40.0*np.pi/180.0, 0.0, 0.0], "crashed": False}
}


@pytest.mark.parametrize("crash_type", crash_type)
def test_crash(crash_type):
    
    crash_dict = spec[crash_type]
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/")
    aircraft = Aircraft(
        initial_position=crash_dict["position"],
        initial_velocity=crash_dict["velocity"],
        initial_attitude=crash_dict["attitude"],
        data_path=path
    )
    assert crash_dict["crashed"] == aircraft.crashed
