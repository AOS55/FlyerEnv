import pytest
from pyflyer import Aircraft

def test_trim():
    aircraft = Aircraft()
    trim = aircraft.trim(1000.0, 100.0, 100)
    print(f"Trim_Result: {trim}")

if __name__=="__main__":
    trim = test_trim()
