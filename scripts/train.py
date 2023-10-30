import gymnasium as gym
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback


class Workspace:

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        exp_name = '_'.join([
            cfg.env_name, cfg.agent_type, str(cfg.seed)
        ])

        if cfg.use_wandb:
            self.run = wandb.init(project="flyer-train", group=cfg.env_name, config=OmegaConf.to_container(cfg, resolve=True), sync_tensorboard=True)

        if cfg.env_config:
            self.train_env = make_vec_env(cfg.env_name, n_envs=cfg.n_envs, seed=cfg.seed, env_kwargs={"config": cfg.env_config})
            self.eval_env = gym.make(cfg.env_name, config=cfg.env_config, render_mode="rgb_array")
        else:
            self.train_env = make_vec_env(cfg.env_name, n_envs=cfg.n_envs, seed=cfg.seed)
            self.eval_env = gym.make(cfg.env_name)

        self.eval_env = Monitor(self.eval_env)

        self.eval_callback = EvalCallback(self.eval_env,
                                          best_model_save_path=f"./logs/{exp_name}",
                                          eval_freq=cfg.eval_freq,
                                          deterministic=True,
                                          render=False)
        
        self.model = SAC(
            "MlpPolicy",
            self.train_env,
            verbose=1,
            tensorboard_log=f".runs/sac"
        )

        return

    def train(self):

        if self.cfg.use_wandb:
            callback = [WandbCallback(
                            model_save_path=f"{self.work_dir}/{self.run.id}",
                                verbose=2
                            ), self.eval_callback
                        ]
        else:
            callback = [self.eval_callback]


        self.model.learn(total_timesteps=self.cfg.total_timesteps,
                         log_interval=self.cfg.log_interval,
                         progress_bar=True,
                         callback=callback)

        if self.cfg.use_wandb:
            self.run.finish()

        return
    
    def load_snapshot(self):
        return

@hydra.main(config_path='conf/.', config_name='control')
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    # root_dir = Path.cwd()
    # snapshot = root_dir / 'snapshot.pt'
    # if snapshot.exists():
    #     print(f'resuming: {snapshot}')
    #     workspace.load_snapshot()
    workspace.train()


if __name__=="__main__":
    main()