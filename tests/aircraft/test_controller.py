import pytest
from world import Aircraft

def test_control():
    aircraft = Aircraft()
    aircraft.reset([0.0, 0.0, -1000.0],
                   0.0, 
                   100.0)
