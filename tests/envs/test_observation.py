import pytest

from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.observation import observation_factory

spec = ["Dynamics"]

@pytest.mark.parametrize("obs_spec", spec)
def test_observation(obs_spec):
    config = {"type": obs_spec}
    observation_factory(AbstractEnv, config)
