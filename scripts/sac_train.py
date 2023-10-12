import gymnasium as gym
from stable_baselines3 import SAC


def main(): 


    env_config = {
        "area": (256, 256)
    }
    env = gym.make("trajectory-v1", config=env_config)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000, log_interval=4, progress_bar=True)

    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()


if __name__=="__main__":
    main()
