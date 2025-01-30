import gymnasium as gym
import flyer_env
import time
import numpy as np

flyer_env.register_flyer_envs()

def main():
    env = gym.make("flyer_control-v1",
        control_type="altitude",
        target_value=500.0,
        tolerance=10.0,
        env_config={"max_episode_steps": 10, "steps_per_action": 5}
    )


    env.reset(seed=5)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


if __name__ == "__main__":
    main()
