from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import flyer_env
import os
import logging
import wandb
import json
import numpy as np
from typing import List, Dict
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
        # self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

        # Setup logging
        log_file_path = os.path.join(log_dir, 'training.log')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
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
        summary_file_path = os.path.join(self.log_dir, 'metrics_summary.json')
        with open(summary_file_path, 'w') as f:
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

def create_callbacks(log_dir: str, cfg: Dict, eval_env: gym.Env) -> CallbackList:
    """Create all callbacks for training"""
    metrics_callback = MetricsCallback(log_dir, use_wandb=cfg.use_wandb)
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callbacks.checkpoint_freq,  # Changed from cfg.checkpoint_freq
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix="sac_model"
    )
    eval_plotting_callback = EvalPlottingCallback(
        eval_freq=cfg.callbacks.eval_freq,  # Changed from cfg.get('eval_freq', 10000)
        log_dir=log_dir,
        n_eval_episodes=cfg.callbacks.n_eval_episodes  # Changed from cfg.get('n_eval_episodes', 3)
    )
    eval_callback_sb3 = EvalCallback(
        eval_env=eval_env,                  # The separate environment for evaluation
        best_model_save_path=log_dir,       # Save 'best_model.zip' here
        log_path=log_dir,                   # Save evaluation logs (evaluations.npz) here
        eval_freq=cfg.callbacks.eval_freq,  # How often to run evaluation (in steps)
                                            # Ensure this frequency makes sense for eval cost
        n_eval_episodes=cfg.callbacks.n_eval_episodes, # Number of episodes per evaluation
        deterministic=True,                 # Use deterministic actions for evaluation
        render=False                        # Don't render evaluation visually
    )


    return CallbackList([metrics_callback, checkpoint_callback, eval_plotting_callback, eval_callback_sb3])

# def setup_logging() -> str:
#     """Create logging directory with timestamp"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_dir = os.path.join("logs", f"run_{timestamp}")
#     os.makedirs(log_dir, exist_ok=True)
#     return log_dir

@hydra.main(version_base="1.1", config_path="configs") # Keep config_name removed
def train(cfg: DictConfig):
    # Setup logging directory

    # --- Add Debug Prints (Optional but recommended from previous step) ---
    print("\n" + "="*20 + " HYDRA CONFIG DEBUG " + "="*20)
    print("--- Raw OmegaConf Object (YAML representation) ---")
    try:
        print(OmegaConf.to_yaml(cfg))
    except Exception as e:
        print(f"Error printing OmegaConf object: {e}")
    # ... (add the specific value checks too if helpful) ...
    print("="*60 + "\n")
    # --- End Debug Prints ---


    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)
        # Set Gymnasium global seed too
        gym.utils.seeding.np_random(cfg.seed)


    # Save full config
    OmegaConf.save(cfg, 'resolved_config.yaml')

    # Initialize wandb if enabled
    if cfg.use_wandb:
        try:
            wandb.init(
                project=cfg.project_name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=cfg.get('run_name', None), # Use run_name if provided
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )
            logging.info(f"WandB initialized for project '{cfg.project_name}', run '{wandb.run.name}'")
        except ImportError:
            logging.error("wandb package not found, but use_wandb=True. Please install wandb.")
            cfg.use_wandb = False # Disable wandb if import fails
        except Exception as e:
            logging.error(f"Failed to initialize WandB: {e}")
            cfg.use_wandb = False # Disable wandb if init fails

    # --- Environment Creation ---
    logging.info("Creating training environment...")
    env = None
    eval_env = None
    try:
        env_id = cfg.env.name
        # Resolve env params before passing to gym.make
        env_params_dict = OmegaConf.to_container(cfg.env.params, resolve=True) if hasattr(cfg.env, 'params') else {}
        env = gym.make(env_id, **env_params_dict)
        logging.info(f"Environment '{env_id}' created successfully.")
        logging.info(f"Observation Space: {env.observation_space}")
        logging.info(f"Action Space: {env.action_space}")
        logging.info("Creating evaluation environment...")
        eval_env = gym.make(env_id, **env_params_dict)
        logging.info(f"Evaluation environment '{env_id}' created successfully.")

    except Exception as e:
        logging.error(f"Failed environment creation: gym.make(ID='{cfg.env.get('name', 'N/A')}'): {e}")
        import traceback
        traceback.print_exc()
        if env: env.close()
        if cfg.get('use_wandb', False) and wandb is not None and wandb.run:
             wandb.finish(exit_code=1)
        return # Exit training

    # --- HER Integration ---
    replay_buffer_class = None
    replay_buffer_kwargs = {}
    policy_class = "MlpPolicy" # Default policy

    use_her = cfg.agent.get("use_her", False)

    if use_her:
        logging.info("HER requested via configuration.")

        # 1. Check if the environment observation space is a dictionary (required for HER)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            logging.error("HER requires a Dict observation space (containing 'observation', 'achieved_goal', 'desired_goal'). Disabling HER.")
            use_her = False
        else:
            # 2. Check if the necessary keys are in the observation space
            required_keys = {"observation", "achieved_goal", "desired_goal"}
            if not required_keys.issubset(env.observation_space.spaces.keys()):
                logging.error(f"Observation space Dict missing required keys for HER ({required_keys}). Disabling HER.")
                use_her = False
            else:
                 policy_class = "MultiInputPolicy" # Use MultiInputPolicy for Dict observations
                 logging.info(f"Using '{policy_class}' for HER with Dict observation space.")

    if use_her: # Check again in case it was disabled by checks
        replay_buffer_class = HerReplayBuffer

        # 3. Determine max_episode_length (required by HerReplayBuffer)
        max_episode_length = None
        if env.spec is not None and env.spec.max_episode_steps is not None:
            max_episode_length = env.spec.max_episode_steps
            logging.info(f"HER: Using max_episode_length from env.spec: {max_episode_length}")
        elif cfg.env.params.get("max_episode_steps") is not None:
            max_episode_length = cfg.env.params.max_episode_steps
            logging.info(f"HER: Using max_episode_length from config env.params: {max_episode_length}")

        if max_episode_length is None:
            logging.error("HER requires 'max_episode_steps' to be defined either in the environment spec or config env.params. Disabling HER.")
            replay_buffer_class = None # Disable HER if length is missing
            policy_class = "MlpPolicy" # Revert policy if HER disabled
        else:
            # 4. Prepare HerReplayBuffer keyword arguments
            her_kwargs = {
                "env": env,
                "buffer_size": cfg.agent.buffer_size, # Use original buffer size from config
                "max_episode_length": max_episode_length,
                "goal_selection_strategy": cfg.agent.get("goal_selection_strategy", "future"), # Get from config or use default
                "n_sampled_goal": cfg.agent.get("n_sampled_goal", 4), # Get from config or use default
                "online_sampling": True, # SAC typically uses online sampling
                "device": "auto", # Or get from cfg.agent if specified
            }
            # Only merge if HER is actually enabled
            if replay_buffer_class is HerReplayBuffer:
                 replay_buffer_kwargs.update(her_kwargs)
                 logging.info(f"HER replay buffer configured with kwargs: {replay_buffer_kwargs}")


    # --- End HER Integration ---

    # Prepare SAC agent parameters (remove HER-specific keys)
    agent_params = OmegaConf.to_container(cfg.agent, resolve=True)
    agent_params.pop("use_her", None)
    agent_params.pop("goal_selection_strategy", None)
    agent_params.pop("n_sampled_goal", None)
    # Make sure buffer_size is not passed if HER is used (it's in replay_buffer_kwargs)
    if replay_buffer_class is HerReplayBuffer:
        agent_params.pop("buffer_size", None)


    # Initialize SAC agent
    logging.info(f"Initializing SAC agent with policy '{policy_class}'...")
    model = None
    try:
        model = SAC(
            policy=policy_class,
            env=env,
            replay_buffer_class=replay_buffer_class,    # None if HER is not used
            replay_buffer_kwargs=replay_buffer_kwargs, # Empty if HER is not used
            verbose=1,                                 # Set verbosity
            seed=cfg.seed,                             # Pass seed
            **agent_params                            # Pass the rest of SAC params
        )
        logging.info("SAC model initialized successfully.")
        logging.info(f"Using Replay Buffer: {model.replay_buffer.__class__.__name__}")

    except Exception as e:
        logging.error(f"Failed to initialize SAC model: {e}")
        import traceback
        traceback.print_exc()
        if cfg.get('use_wandb', False) and wandb is not None and wandb.run:
             wandb.finish(exit_code=1)
        env.close()
        return # Exit training


    # Setup callbacks
    log_dir_path = os.getcwd()
    callbacks_list = create_callbacks(log_dir_path, cfg, eval_env) # Use your existing function

    # Load from checkpoint if specified
    # ... (Your checkpoint loading logic) ...


    # Train the agent
    logging.info(f"Starting training for {cfg.total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks_list,
            reset_num_timesteps=not cfg.get('resume_training', False), # Check resume flag
            log_interval=10 # Log training stats frequency (episodes)
        )
        logging.info("Training finished.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model regardless of training success/failure
        final_model_path = 'final_model.zip'
        model.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        # Close environment
        if env:
            env.close()
            logging.info("Training environment closed.")
        if eval_env:
            eval_env.close()
            logging.info("Evaluation environment closed.")

        if cfg.use_wandb and wandb.run:
            wandb.finish()
            logging.info("WandB run finished.")

if __name__ == "__main__":
    # Ensure environments are registered before Hydra parses config
    flyer_env.register_flyer_envs()
    train()
