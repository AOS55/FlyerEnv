import gymnasium as gym
from stable_baselines3 import SAC


def main(): 
    env = gym.make("flyer-v1")

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000, log_interval=4)

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


if __name__=="__main__":
    main()
