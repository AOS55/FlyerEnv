import gymnasium as gym
import flyer_env
flyer_env.register_flyer_envs()

def main():
    env = gym.make("flyer_control-v1",
        control_type="altitude",
        target_value=500.0,
        tolerance=10.0
    )

    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Your control policy here
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
