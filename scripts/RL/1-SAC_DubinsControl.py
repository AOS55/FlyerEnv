from stable_baselines3 import SAC
import hydra
from omegaconf import DictConfig
import gymnasium as gym
import flyer_env
import os
from datetime import datetime
import logging

def setup_logging() -> str:
    # Create logs directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@hydra.main(version_base="1.1", config_path="configs", config_name="dubins_control")
def train(cfg: DictConfig):
    # Setup basic logging
    log_dir = setup_logging()

    # Create environment
    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        seed=cfg.seed,
        control_type=cfg.control_type,
        target_value=cfg.target_value,
        simplified_spaces=True,
        tolerance=cfg.tolerance
    )

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

    # Train the agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save the trained model
    model.save(os.path.join(log_dir, 'final_model'))

if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    train()
