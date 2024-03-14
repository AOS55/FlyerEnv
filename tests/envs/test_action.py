import pytest

from flyer_env.envs.common.abstract import AbstractEnv
from flyer_env.envs.common.action import action_factory

spec = ["ContinuousAction", "ControlledAction", "PursuitAction"]


@pytest.mark.parametrize("act_spec", spec)
def test_action(act_spec):
    config = {"type": act_spec}
    action_factory(AbstractEnv, config)
