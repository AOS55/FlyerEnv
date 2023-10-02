import gymnasium as gym
import numpy as np

def main():

    env_config = {
        "duration": 10.0,
        "action": {
                "type": "PursuitAction"
            }
    }

    env = gym.make("flyer-v1", config=env_config, render_mode="rgb_array")

    obs, info = env.reset()
    observations = []
    terminated = truncated = False
    goal = env.unwrapped.goal

    while not (terminated or truncated):
        action = {"goal_pos": goal[0:2], "other_controls": np.array([goal[-1]/10000, 100.0/300.0])}
        obs, reward, terminated, truncated, info = env.step(action)
    env.close()

if __name__=="__main__":
    main()
