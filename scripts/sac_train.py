import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

def main(): 


    env_config = {
        "area": (256, 256),
        "simulation_frequency": 100.0
    }

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_name": "trajectory-v1",
        "env_config": env_config
    }

    # env_id = "trajectory-v1"
    # env = gym.make("trajectory-v1", config=env_config)

    run = wandb.init(
        project="FlyerEnv-tests",
        config=config,
        sync_tensorboard=False
    )

    env = make_vec_env(config["env_name"], n_envs=256, seed=0, env_kwargs={"config": env_config})

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=config["total_timesteps"],
                log_interval=4,
                progress_bar=True,
                callback=WandbCallback(
                    model_save_path=f"models/{run.id}",
                    verbose=2
                ))
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
