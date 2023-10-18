import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback


def main():

    env_config = {
        "area": (256, 256),
        "simulation_frequency": 100.0,
        "observation": {
            "type": "Trajectory"
        },
        "duration": 10.0,
        "trajectory_config": {
            "name": "sl"
        },
    }

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 40000000,
        "env_name": "trajectory-v1",
        "env_config": env_config
    }

    env = gym.make(config["env_name"], config=env_config)
    # env = gym.make(config["env_name"])

    run = wandb.init(
        project="FlyerEnv-tests",
        config=config,
        sync_tensorboard=True
    )

    env = make_vec_env(config["env_name"], n_envs=32, seed=0, env_kwargs={"config": env_config})
    eval_env = gym.make(config['env_name'], config=env_config)
    eval_env = Monitor(eval_env)
    # env = make_vec_env(config["env_name"], n_envs=32)
    # env = gym.wrappers.RecordEpisodeStatistics(env)

    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)
    

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f".runs/sac")
    model.learn(total_timesteps=config["total_timesteps"],
                log_interval=4,
                progress_bar=True,
                callback=[WandbCallback(
                    model_save_path=f"models/{run.id}",
                    verbose=2
                ), eval_callback])
    run.finish()
    
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True


if __name__=="__main__":
    main()
