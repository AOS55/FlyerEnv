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
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if not obs_history:
            print("Warning: No observation history to save.")
            return
        # Make sure history is not empty before trying to get headers/values
        if not any(obs_history.values()):
             print("Warning: Observation history is empty.")
             return
        headers = list(obs_history.keys())
        writer.writerow(headers)

        # Find the length of the first non-empty list to determine number of rows
        num_rows = 0
        for key in headers:
             if obs_history[key]:
                  num_rows = len(obs_history[key])
                  break # Found a non-empty list

        if num_rows == 0:
             print("Warning: All observation history lists are empty.")
             return

        for i in range(num_rows):
            row = []
            for key in headers:
                 # Check if key exists and index is valid
                 if key in obs_history and i < len(obs_history[key]):
                      row.append(obs_history[key][i])
                 else:
                      row.append(None) # Append None if key missing or index out of bounds
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

    # --- ADD DEBUGGING ---
    print("-" * 20 + " DEBUG INFO " + "-" * 20)
    try:
        # import os # os is already imported at the top
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Received run_dir        : {cfg.run_dir}")
        print(f"Constructed config path : {training_config_path}")
        # Use resolve() to get the absolute path *before* checking existence
        abs_config_path = training_config_path.resolve()
        print(f"Absolute config path    : {abs_config_path}")
        # Now check existence using the original (potentially relative) path object
        print(f"Path object exists?     : {training_config_path.exists()}")
        # Optional: Check parent directory
        parent_dir = training_config_path.parent
        print(f"Parent dir path         : {parent_dir}")
        print(f"Parent dir exists?      : {parent_dir.exists()}")
        print(f"Is parent a directory?  : {parent_dir.is_dir()}")
    except Exception as e:
        print(f"Error during debug printing: {e}")
    print("-" * 52)
    # --- END DEBUGGING ---

    if not training_config_path.exists():
        print(f"Error: Training config not found at {training_config_path}")
        print(f"(Absolute path checked: {abs_config_path})") # Also print absolute path in error
        print("Ensure 'run_dir' points to a valid Hydra output directory and the script is run from the project root.")
        return

    print(f"Loading training config from: {training_config_path}")
    try:
        training_cfg = OmegaConf.load(training_config_path)
    except Exception as e:
        print(f"Error loading OmegaConf from {training_config_path}: {e}")
        return


    # --- Prepare Paths ---
    # Ensure model_path uses the potentially relative run_dir correctly
    model_path = Path(cfg.run_dir) / cfg.model_filename
    # Ensure output_dir uses the potentially relative run_dir correctly
    output_dir = Path(cfg.run_dir) / cfg.output_subdir
    csv_filename = f"evaluation_results_{Path(cfg.model_filename).stem}.csv"
    output_csv_path = output_dir / csv_filename

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print(f"(Absolute path checked: {model_path.resolve()})")
        return

    print(f"Loading trained model from: {model_path}")

    # --- Instantiate Environment using Training Config --- #
    env = None # Initialize env
    try:
        # Important: Use training config for env params, but eval config for render mode
        # Make sure training_cfg.env exists before accessing params
        if not hasattr(training_cfg, 'env') or not hasattr(training_cfg.env, 'params'):
             print("Error: training_cfg does not contain expected 'env.params' structure.")
             return

        env_params = training_cfg.env.params
        # Use OmegaConf.merge to safely add/override render_mode
        # Ensure cfg.render_mode exists
        render_mode_override = {'render_mode': cfg.get('render_mode', 'rgb_array')} # Default if missing
        env_params = OmegaConf.merge(env_params, render_mode_override)
        # Convert back to standard dict for gym.make
        env_params_dict = OmegaConf.to_container(env_params, resolve=True)

        # Ensure training_cfg.env.name exists
        if not hasattr(training_cfg.env, 'name'):
             print("Error: training_cfg does not contain expected 'env.name'.")
             return

        print(f"Creating environment: {training_cfg.env.name}")
        env = gym.make(training_cfg.env.name, **env_params_dict)

    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Check if the environment name and parameters in the loaded training config are correct.")
        if env: env.close() # Close env if partially created before error
        return

    # --- Load Model --- #
    model = None # Initialize model
    try:
        # TODO: Make agent loading configurable based on training_cfg.agent._target_?
        # For now, assuming SAC as in the original script
        model = SAC.load(model_path, env=env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        if env: env.close() # Close env if model loading fails
        return

    # --- Run Rollouts --- #
    all_episodes_history = [] # Store history for each episode
    print(f"Running {cfg.num_episodes} evaluation episodes...")

    try: # Add try block around rollouts
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
                # Ensure training_cfg.env.params exists and get max_episode_steps safely
                max_steps = float('inf')
                if hasattr(training_cfg, 'env') and hasattr(training_cfg.env, 'params'):
                     max_steps = training_cfg.env.params.get('max_episode_steps', float('inf'))

                if step_count >= max_steps:
                    truncated = True # Manually truncate if step limit reached

            print(f"Episode {episode + 1} finished after {step_count} steps. Terminated: {terminated}, Truncated: {truncated}")
            all_episodes_history.append(episode_history)

    except Exception as e:
        print(f"Error during evaluation rollout: {e}")
        import traceback
        traceback.print_exc()
    finally: # Ensure environment is closed even if rollouts fail
        if env:
            env.close()
            print("Environment closed.")

    # --- Aggregate and Save Results --- #
    # Combine histories - simple approach: concatenate lists
    # More sophisticated analysis might average or plot distributions
    aggregated_history = {}
    if all_episodes_history:
        # Find all unique keys across all episodes first
        all_keys = set()
        for episode_hist in all_episodes_history:
             all_keys.update(episode_hist.keys())

        # Initialize aggregated_history with all keys
        for key in all_keys:
             aggregated_history[key] = []

        # Concatenate data from all episodes
        for episode_hist in all_episodes_history:
            # Get length of one of the lists in this episode (assume lengths are consistent within episode)
            episode_len = 0
            if episode_hist:
                 episode_len = len(next(iter(episode_hist.values()), []))

            for key in all_keys:
                if key in episode_hist:
                    aggregated_history[key].extend(episode_hist[key])
                else:
                    # Handle case where a key might be missing in an episode
                    aggregated_history[key].extend([None] * episode_len)

    if aggregated_history:
        save_to_csv(aggregated_history, output_csv_path)
    else:
        print("No data collected during evaluation episodes.")


if __name__ == "__main__":
    # Register environments if necessary (usually done in train.py's import)
    # Ensure it's safe to call multiple times or handle registration appropriately
    try:
        flyer_env.register_flyer_envs()
    except Exception as e:
         print(f"Note: Environment registration might have already occurred. Error: {e}")
    run_evaluation()
