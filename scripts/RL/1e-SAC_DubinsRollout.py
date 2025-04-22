import gymnasium as gym
import flyer_env
import numpy as np
import csv
import os
import argparse
from stable_baselines3 import SAC
from pathlib import Path

def save_to_csv(obs_history, control_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{control_type}_sac_rollout.csv")

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = list(obs_history.keys())
        writer.writerow(headers)

        for i in range(len(next(iter(obs_history.values())))):
            row = [obs_history[key][i] for key in headers]
            writer.writerow(row)

    print(f"Data saved to {file_path}")

def rollout_agent(model_path, output_dir, control_type, target_value, tolerance, num_steps=1000):
    print(f"Loading trained SAC model from {model_path}")

    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        control_type=control_type,
        target_value=target_value,
        tolerance=tolerance
    )

    model = SAC.load(model_path, env=env)
    obs, _ = env.reset()
    obs_history = {}

    for _ in range(num_steps):
        obs_dict = env.unwrapped._observation.to_dict(obs)
        action, _ = model.predict(obs, deterministic=True)
        act_dict = env.unwrapped._action.to_dict(action)
        obs, reward, truncated, terminated, _ = env.step(action)

        for key in obs_dict:
            if key not in obs_history:
                obs_history[key] = []
            obs_history[key].append(obs_dict[key])

        for key in act_dict:
            if key not in obs_history:
                obs_history[key] = []
            obs_history[key].append(act_dict[key])

        if 'reward' not in obs_history:
            obs_history['reward'] = []
        obs_history['reward'].append(reward)

        if terminated or truncated:
            break

    save_to_csv(obs_history, control_type, output_dir)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained SAC model file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV output")
    parser.add_argument("--control_type", type=str, choices=["altitude", "heading", "speed"], required=True, help="Control type to evaluate")
    parser.add_argument("--target_value", type=float, required=True, help="Target value for control type")
    parser.add_argument("--tolerance", type=float, default=10.0, help="Tolerance for the control task")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of rollout steps")
    args = parser.parse_args()

    flyer_env.register_flyer_envs()
    rollout_agent(args.model_path, args.output_dir, args.control_type, args.target_value, args.tolerance, args.num_steps)
