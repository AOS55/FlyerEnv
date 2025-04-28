import gymnasium as gym
import flyer_env
import numpy as np
import csv
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC
from pathlib import Path

def save_to_csv(obs_history, output_path):
    """Saves the rollout history to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if not obs_history:
            print("Warning: No observation history to save.")
            return
        headers = list(obs_history.keys())
        writer.writerow(headers)

        # Find the length of the first list to determine number of rows
        num_rows = len(next(iter(obs_history.values())))
        for i in range(num_rows):
            row = [obs_history[key][i] if i < len(obs_history[key]) else None for key in headers]
            writer.writerow(row)

    print(f"Evaluation data saved to {output_path}")

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def run_evaluation(cfg: DictConfig) -> None:
    """Loads a trained model and runs evaluation episodes."""
    print("Evaluation Configuration:")
    print(OmegaConf.to_yaml(cfg))

    if cfg.run_dir == '???':
        print("Error: Please specify the 'run_dir' pointing to the training output directory.")
        print("Example: python scripts/RL/eval.py run_dir=outputs/YYYY-MM-DD/HH-MM-SS")
        return

    # --- Load Original Training Configuration ---
    training_config_path = Path(cfg.run_dir) / ".hydra" / "config.yaml"
    if not training_config_path.exists():
        print(f"Error: Training config not found at {training_config_path}")
        print("Ensure 'run_dir' points to a valid Hydra output directory.")
        return

    print(f"Loading training config from: {training_config_path}")
    training_cfg = OmegaConf.load(training_config_path)

    # --- Prepare Paths ---
    model_path = Path(cfg.run_dir) / cfg.model_filename
    output_dir = Path(cfg.run_dir) / cfg.output_subdir
    csv_filename = f"evaluation_results_{Path(cfg.model_filename).stem}.csv"
    output_csv_path = output_dir / csv_filename

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading trained model from: {model_path}")

    # --- Instantiate Environment using Training Config --- #
    # Important: Use training config for env params, but eval config for render mode
    env_params = training_cfg.env.params
    # Use OmegaConf.merge to safely add/override render_mode
    env_params = OmegaConf.merge(env_params, {'render_mode': cfg.render_mode})
    # Convert back to standard dict for gym.make
    env_params_dict = OmegaConf.to_container(env_params, resolve=True)

    print(f"Creating environment: {training_cfg.env.name}")
    try:
        env = gym.make(training_cfg.env.name, **env_params_dict)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Check if the environment name and parameters in the loaded training config are correct.")
        return

    # --- Load Model --- #
    # TODO: Make agent loading configurable based on training_cfg.agent._target_?
    # For now, assuming SAC as in the original script
    try:
        model = SAC.load(model_path, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # --- Run Rollouts --- #
    all_episodes_history = [] # Store history for each episode
    print(f"Running {cfg.num_episodes} evaluation episodes...")

    for episode in range(cfg.num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_history = {} # History for the current episode
        step_count = 0

        while not terminated and not truncated:
            # Record observations/actions BEFORE the step
            obs_dict = getattr(env.unwrapped, '_observation', None)
            if obs_dict and hasattr(obs_dict, 'to_dict'):
                current_obs_dict = obs_dict.to_dict(obs)
                for key, value in current_obs_dict.items():
                    if key not in episode_history: episode_history[key] = []
                    episode_history[key].append(value)
            else:
                 # Fallback if observation structure is different
                 if 'observation' not in episode_history: episode_history['observation'] = []
                 episode_history['observation'].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)

            action, _ = model.predict(obs, deterministic=cfg.deterministic)

            act_dict_obj = getattr(env.unwrapped, '_action', None)
            if act_dict_obj and hasattr(act_dict_obj, 'to_dict'):
                 current_act_dict = act_dict_obj.to_dict(action)
                 for key, value in current_act_dict.items():
                     if key not in episode_history: episode_history[key] = []
                     episode_history[key].append(value)
            else:
                 # Fallback if action structure is different
                 if 'action' not in episode_history: episode_history['action'] = []
                 episode_history['action'].append(action.tolist() if isinstance(action, np.ndarray) else action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Record reward AFTER the step
            if 'reward' not in episode_history: episode_history['reward'] = []
            episode_history['reward'].append(reward)

            if cfg.render_mode == 'human':
                env.render()

            # Check against training max_episode_steps if available
            max_steps = training_cfg.env.params.get('max_episode_steps', float('inf'))
            if step_count >= max_steps:
                truncated = True # Manually truncate if step limit reached

        print(f"Episode {episode + 1} finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")
        all_episodes_history.append(episode_history)

    env.close()
    print("Environment closed.")

    # --- Aggregate and Save Results --- #
    # Combine histories - simple approach: concatenate lists
    # More sophisticated analysis might average or plot distributions
    aggregated_history = {}
    if all_episodes_history:
        # Initialize keys from the first episode
        for key in all_episodes_history[0].keys():
            aggregated_history[key] = []
        # Concatenate data from all episodes
        for episode_hist in all_episodes_history:
            for key in aggregated_history.keys():
                if key in episode_hist:
                    aggregated_history[key].extend(episode_hist[key])
                else:
                     # Handle case where a key might be missing in later episodes (unlikely but safe)
                     aggregated_history[key].extend([None] * len(next(iter(episode_hist.values())))) # Extend with Nones

    if aggregated_history:
        save_to_csv(aggregated_history, output_csv_path)
    else:
        print("No data collected during evaluation episodes.")

if __name__ == "__main__":
    # Register environments if necessary (already done in train.py)
    flyer_env.register_flyer_envs()
    run_evaluation()
