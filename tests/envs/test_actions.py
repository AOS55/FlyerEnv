import pytest
import gymnasium as gym

action_configs = [
    {"type": "ContinuousAction"},
    {"type": "LongitudinalAction"},
    {"type": "HeadingAction"},
    {"type": "ControlledAction"},
    {"type": "PursuitAction"}
]


@pytest.mark.parametrize("action_config", action_configs)
def test_action_type(action_config): 
    env = gym.make("flyer-v1")
    env.unwrapped.configure({"action": action_config})
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(obs)
    env.close()
