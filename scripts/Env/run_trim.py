import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict

from flyer_env.envs.common.single_agent_env import SingleAgentEnv

def main():
    """
    Simple script to demonstrate a StraightAndLevel trim condition using direct observation plotting.
    """
    print("\n=== Testing StraightAndLevel trim condition ===")

    # Create configuration for a straight and level flight
    config = {
        "time_step": 1/60,
        "max_episode_steps": 1000,
        "aircraft_config": [{
            "type": "full",
            "config": {
                "ac_type": "twin_otter",
                "action_config": {
                    "normalize": False
                }
            },
            "start_config": {
                "type": "fixed",
                "config": {
                    "position": [0.0, 0.0, -0.0],  # negative altitude in Flyer
                    "speed": 80.0,
                    "heading": 0.0
                }
            },
            "trim_condition": {
                "condition_type": "StraightAndLevel",
                "airspeed": 80.0
            },
            "action_type": "Continuous",
            "observation_type": "Continuous",
            "task_config": {
                "type": "Control",
                "config": {
                    "control_type": "Altitude",
                    "target": 0.0,
                    "tolerance": 10.0
                }
            }
        }]
    }

    # Create environment
    env = SingleAgentEnv(config=config, render_mode="rgb_array")

    # Reset environment
    obs, info = env.reset()

    # Initialize arrays to store data
    time_points = []
    x_positions = []
    z_positions = []
    velocities = []
    roll_angles = []
    pitch_angles = []

    # Run simulation and collect data
    dt = config['time_step']
    time_elapsed = 0.0
    num_steps = 10000

    # Observation array structure: [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
    # Record initial state
    time_points.append(time_elapsed)
    x_positions.append(obs[0])
    z_positions.append(-obs[2])  # Convert to altitude (positive up)

    # Get velocities from indices 6, 7, 8 (u, v, w)
    velocities.append(np.sqrt(obs[6]**2 + obs[7]**2 + obs[8]**2))  # Calculate airspeed

    # Get roll and pitch from indices 3 and 4
    roll_angles.append(np.degrees(obs[3]))
    pitch_angles.append(np.degrees(obs[4]))

    print(f"Starting simulation for {num_steps} steps...")
    for i in range(num_steps):
        # Zero input to maintain trim condition
        action = np.zeros(env.action_space.shape)
        action = np.array([-0.028833359, 0.0, 0.77409241, 0.0])

        # Step the environment
        obs, reward, term, trunc, info = env.step(action)

        if i < 3:  # Print the first few observations to verify format
            print(f"Step {i} observation: {obs}")

        time_elapsed += dt

        # Store data with correct indices
        time_points.append(time_elapsed)
        x_positions.append(obs[0])
        z_positions.append(-obs[2])  # Convert to altitude (positive up)

        # Get velocities from indices 6, 7, 8 (u, v, w)
        velocities.append(np.sqrt(obs[6]**2 + obs[7]**2 + obs[8]**2))  # Calculate airspeed

        # Get roll and pitch from indices 3 and 4
        roll_angles.append(np.degrees(obs[3]))
        pitch_angles.append(np.degrees(obs[4]))

        # Print status occasionally
        if i % 50 == 0:
            print(f"Step {i}/{num_steps}, airspeed: {velocities[-1]:.2f} m/s")

        # if term or trunc:
        #     print("Episode ended early")
        #     break

    env.close()
    print("Simulation complete")

    # Create a single graph with multiple subplots
    plt.figure(figsize=(12, 8))

    # Position plot
    plt.subplot(2, 2, 1)
    plt.plot(time_points, x_positions, label='X Position')
    plt.plot(time_points, z_positions, label='Altitude')
    plt.title('Aircraft Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)

    # Velocity plot
    plt.subplot(2, 2, 2)
    plt.plot(time_points, velocities)
    plt.title('Airspeed')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    # Attitude plot
    plt.subplot(2, 2, 3)
    plt.plot(time_points, roll_angles, label='Roll')
    plt.plot(time_points, pitch_angles, label='Pitch')
    plt.title('Aircraft Attitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True)

    # Forward position vs time
    plt.subplot(2, 2, 4)
    plt.plot(time_points, x_positions)
    plt.title('Forward Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('trim_straight_and_level.png')
    print("Saved plot to trim_straight_and_level.png")
    plt.show()

if __name__ == "__main__":
    main()
