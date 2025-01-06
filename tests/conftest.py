import pytest
from pathlib import Path

def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--render",
        action="store_true",
        default=False,
        help="Enable environment rendering during tests"
    )

@pytest.fixture
def render_mode(pytestconfig):
    """Get render mode from command line option."""
    return "human" if pytestconfig.getoption("render") else None

@pytest.fixture
def test_configs_path() -> Path:
    """Path to test configuration files."""
    return Path(__file__).parent / "configs"