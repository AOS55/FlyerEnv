from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import flyer_env
import os
from datetime import datetime
import logging
import wandb
import json
import numpy as np
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import random
import torch


class EvalPlottingCallback(BaseCallback):
    """
    Callback for evaluating and plotting agent behavior periodically during training.
    """
    def __init__(
        self,
        eval_freq: int,
        log_dir: str,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

        # Create plots directory
        self.plots_dir = os.path.join(log_dir, 'eval_plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize storage for evaluation data
        self.eval_data = {
            'obs': [],
            'actions': [],
            'rewards': []
        }

    def _on_step(self) -> bool:
        """
        Check if we should run evaluation and plotting
        """
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
            self._generate_plots()
            # Clear data after plotting
            self.eval_data = {k: [] for k in self.eval_data.keys()}
        return True

    def _run_evaluation(self):
        """
        Run evaluation episodes and collect data
        """
        print(f"\nRunning evaluation at step {self.n_calls}")

        for episode in range(self.n_eval_episodes):
            obs = self.training_env.reset()
            # Handle both single value and tuple return types
            if isinstance(obs, tuple):
                obs = obs[0]
            done = False
            episode_obs = []
            episode_actions = []
            episode_rewards = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                next_obs, reward, done, _ = self.training_env.step(action)

                # Store data
                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)

                obs = next_obs

            # Store episode data
            self.eval_data['obs'].append(episode_obs)
            self.eval_data['actions'].append(episode_actions)
            self.eval_data['rewards'].append(episode_rewards)

    def _generate_plots(self):
        """
        Generate and save evaluation plots
        """
        timestamp = self.n_calls // self.eval_freq

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'Evaluation Results - Step {self.n_calls}')

        # Plot observations
        self._plot_trajectories(
            data=[np.array(episode_obs) for episode_obs in self.eval_data['obs']],
            title='Observations over Time',
            ax=axes[0],
            ylabel='Observation Value'
        )

        # Plot actions
        self._plot_trajectories(
            data=[np.array(episode_actions) for episode_actions in self.eval_data['actions']],
            title='Actions over Time',
            ax=axes[1],
            ylabel='Action Value'
        )

        # Plot rewards
        self._plot_trajectories(
            data=[np.array(episode_rewards) for episode_rewards in self.eval_data['rewards']],
            title='Rewards over Time',
            ax=axes[2],
            ylabel='Reward'
        )

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'eval_plot_{timestamp}.png'))
        plt.close()

    def _plot_trajectories(
        self,
        data: List[np.ndarray],
        title: str,
        ax: plt.Axes,
        ylabel: str
    ):
        """
        Helper function to plot trajectories for multiple episodes
        """
        ax.set_title(title)

        for episode_idx, episode_data in enumerate(data):
            episode_data = np.array(episode_data)

            # Handle different shapes of data
            if len(episode_data.shape) > 1:
                # Multiple components (e.g., observation space has multiple dimensions)
                for dim in range(episode_data.shape[1]):
                    ax.plot(
                        episode_data[:, dim],
                        alpha=0.7,
                        label=f'Episode {episode_idx+1} - Dim {dim}',
                        linestyle='-',
                        marker='',
                        markersize=2
                    )
            else:
                # Single component
                ax.plot(
                    episode_data,
                    alpha=0.7,
                    label=f'Episode {episode_idx+1}',
                    linestyle='-',
                    marker='',
                    markersize=2
                )

        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

class MetricsCallback(BaseCallback):
    """Custom callback for logging metrics during training"""

    def __init__(self, log_dir: str, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )

    def _on_step(self) -> bool:
        """Update metrics on each step"""
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        # Update episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            # Store episode metrics
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Log metrics
            metrics = {
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'mean_reward': np.mean(self.episode_rewards[-100:]),  # Rolling mean of last 100 episodes
                'episode_num': len(self.episode_rewards)
            }

            # Log training info if available
            if hasattr(self.model, 'logger'):
                for key, value in self.model.logger.name_to_value.items():
                    metrics[key] = value

            # Log to console and file
            logging.info(f"Step {self.num_timesteps}: {metrics}")

            # Log to W&B if enabled
            if self.use_wandb:
                wandb.log(metrics, step=self.num_timesteps)

            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        """Save final metrics summary"""
        summary = {
            'episode_rewards': {
                'mean': float(np.mean(self.episode_rewards)),
                'std': float(np.std(self.episode_rewards)),
                'min': float(np.min(self.episode_rewards)),
                'max': float(np.max(self.episode_rewards))
            },
            'episode_lengths': {
                'mean': float(np.mean(self.episode_lengths)),
                'std': float(np.std(self.episode_lengths)),
                'min': float(np.min(self.episode_lengths)),
                'max': float(np.max(self.episode_lengths))
            },
            'total_episodes': len(self.episode_rewards),
            'total_timesteps': self.num_timesteps
        }

        # Save summary to file
        with open(os.path.join(self.log_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

def seed_everything(seed_value):
    """Seeds libraries for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # for multi-GPU.
        # Force deterministic algorithms
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"
        torch.use_deterministic_algorithms(True)
        # Might be needed for older PyTorch versions
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seeded everything with: {seed_value}")

def create_callbacks(log_dir: str, cfg: Dict) -> CallbackList:
    """Create all callbacks for training"""
    metrics_callback = MetricsCallback(log_dir, use_wandb=cfg.use_wandb)
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callbacks.checkpoint_freq,  # Changed from cfg.checkpoint_freq
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix="sac_model"
    )
    eval_callback = EvalPlottingCallback(
        eval_freq=cfg.callbacks.eval_freq,  # Changed from cfg.get('eval_freq', 10000)
        log_dir=log_dir,
        n_eval_episodes=cfg.callbacks.n_eval_episodes  # Changed from cfg.get('n_eval_episodes', 3)
    )

    return CallbackList([metrics_callback, checkpoint_callback, eval_callback])

def setup_logging() -> str:
    """Create logging directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@hydra.main(version_base="1.1", config_path="configs", config_name="1b-control_dubins_speed")
def train(cfg: DictConfig):
    # Setup logging directory
    log_dir = setup_logging()

    print(f"cfg: {cfg}")

    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)

    # Save full config
    OmegaConf.save(cfg, os.path.join(log_dir, 'config.yaml'))

    # Initialize wandb if enabled
    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=log_dir,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Create environment
    logging.info("Creating environment...")
    env = None
    try:
        # Validate environment config
        if not hasattr(cfg, 'env'):
            raise ValueError("Environment configuration missing 'env' section")

        env_cfg = cfg.env
        if not hasattr(env_cfg, 'name'):
            raise ValueError("Environment configuration missing 'name' field")

        env_id = env_cfg.name
        env_params_dict = {}
        if hasattr(env_cfg, 'params'):
            env_params_dict = OmegaConf.to_container(env_cfg.params, resolve=True)
        env = gym.make(env_id, **env_params_dict)
    except Exception as e:
        logging.error(f"Failed environment creation: gym.make(ID='{env_cfg.get('name', 'N/A')}'): {e}")
        import traceback
        traceback.print_exc()
        # Clean up WandB if needed
        if cfg.get('use_wandb', False) and wandb is not None and wandb.run:
            wandb.finish(exit_code=1)
        return # Exit training

    # Initialize SAC agent
    agent_params = OmegaConf.to_container(cfg.agent, resolve=True)
    model = SAC(
        "MlpPolicy",
        env,
        **agent_params
    )

    # Setup callbacks
    callbacks = create_callbacks(log_dir, cfg)

    # Load from checkpoint if specified
    if cfg.get('resume_training', False):
        checkpoint_path = Path(log_dir) / 'checkpoints'
        if checkpoint_path.exists():
            checkpoints = list(checkpoint_path.glob("sac_model_*.zip"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                model = SAC.load(str(latest_checkpoint), env=env)
                start_timestep = int(latest_checkpoint.stem.split('_')[-1])
                logging.info(f"Resuming training from checkpoint at step {start_timestep}")

    # Train the agent
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not cfg.get('resume_training', False)
    )

    # Save final model
    model.save(os.path.join(log_dir, 'final_model'))

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    train()
