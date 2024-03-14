import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def make_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, "videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env


config = {"policy": "MlpPolicy", "total_timesteps": 25000}

wandb.init(
    project="FlyerEnv-tests",
    config=config,
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    monitor_gym=True,  # automatically upload gym environements' videos
    save_code=True,
)

env = DummyVecEnv([make_env])
model = PPO(config["policy"], env, verbose=1, tensorboard_log="runs/ppo")
model.learn(total_timesteps=config["total_timesteps"])
wandb.finish()
