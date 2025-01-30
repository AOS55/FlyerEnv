from stable_baselines3 import SAC
import wandb
from wandb.integration.sb3 import WandbCallback
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import flyer_env
import numpy as np
from typing import Dict, Any
import os
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import json
import logging

# Custom callback for detailed training monitoring
class DebugCallback(BaseCallback):
    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.training_stats = {}

    def _on_step(self) -> bool:
        # Log critic and actor losses
        if hasattr(self.model, 'logger'):
            if len(self.model.logger.name_to_value) > 0:
                for key, value in self.model.logger.name_to_value.items():
                    if 'loss' in key.lower():
                        if key not in self.training_stats:
                            self.training_stats[key] = []
                        self.training_stats[key].append(value)

        # Accumulate episode reward
        self.current_episode_reward += self.locals['rewards'][0]

        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.locals['episode_lengths'][0])
            self.current_episode_reward = 0

            # Log episode statistics
            if len(self.episode_rewards) % self.log_freq == 0:
                mean_reward = np.mean(self.episode_rewards[-self.log_freq:])
                mean_length = np.mean(self.episode_lengths[-self.log_freq:])

                if self.verbose > 0:
                    print(f"Episode {len(self.episode_rewards)}")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean length: {mean_length:.2f}")

                # Log to wandb if enabled
                if wandb.run is not None:
                    wandb.log({
                        "mean_episode_reward": mean_reward,
                        "mean_episode_length": mean_length,
                        "episodes": len(self.episode_rewards)
                    })

        return True

    def get_training_stats(self) -> Dict[str, Any]:
        return self.training_stats

def setup_logging(cfg: DictConfig) -> str:
    # Create logs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # Save configuration
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(dict(cfg), f, indent=4)

    return log_dir

@hydra.main(version_base="1.1", config_path="configs", config_name="dubins_control")
def train(cfg: DictConfig):
    # Setup logging
    log_dir = setup_logging(cfg)
    logging.info(f"Starting training with config: {dict(cfg)}")

    # Initialize W&B
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.project_name,
            config=dict(cfg),
            sync_tensorboard=True,
            dir=log_dir
        )

    # Create and wrap environment
    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        seed=cfg.seed,
        control_type=cfg.control_type,
        target_value=cfg.target_value,
        tolerance=cfg.tolerance
    )
    env = Monitor(env, log_dir)

    # Create evaluation environment
    eval_env = gym.make("flyer_control-v1",
        render_mode=None,
        seed=cfg.seed + 100,
        control_type=cfg.control_type,
        target_value=cfg.target_value,
        tolerance=cfg.tolerance
    )
    eval_env = Monitor(eval_env, os.path.join(log_dir, 'eval'))

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        verbose=1,
        tensorboard_log=log_dir
    )

    # Setup callbacks
    debug_callback = DebugCallback(verbose=1, log_freq=1000)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='sac_model'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval_logs'),
        eval_freq=100,
        deterministic=True,
        render=False
    )

    callbacks = [debug_callback, checkpoint_callback, eval_callback]
    if cfg.use_wandb:
        callbacks.append(WandbCallback(gradient_save_freq=100, verbose=2))

    try:
        # Train the agent
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks
        )

        # Save final model and training stats
        model.save(os.path.join(log_dir, 'final_model'))
        training_stats = debug_callback.get_training_stats()
        with open(os.path.join(log_dir, 'training_stats.json'), 'w') as f:
            json.dump(training_stats, f, indent=4)

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise e

    finally:
        if cfg.use_wandb:
            run.finish()

if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    train()
