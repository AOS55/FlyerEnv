import numpy as np
import matplotlib.pyplot as plt
import logging

# Import the base env class AND the necessary config dataclasses/helpers
from flyer_env.envs.common.single_agent_env import SingleAgentEnv
from flyer_env.envs.common.config_defs import (
    SingleAgentEnvConfig, EnvConfig, AircraftDefinition, AircraftPhysicsConfig,
    TaskConfig, ControlTaskConfigData, AgentRenderConfig,
)
from flyer_env.envs.common.config_utils import (
    # Import builders if needed, or define physics manually for testing
    FullAircraftPresetBuilder,
    build_fixed_start_config, # Use helper for start config
)


logging.basicConfig(level=logging.INFO) # Setup basic logging
logger = logging.getLogger(__name__)

def main():
    """
    Simple script to demonstrate a StraightAndLevel trim condition using direct observation plotting.
    Uses the NEW structured configuration.
    """
    print("\n=== Testing StraightAndLevel trim condition ===")

    # --- 1. Build the structured configuration ---

    # Env Base Config
    env_base_config = EnvConfig(
        time_step=1/60,
        max_episode_steps=1000,
        agent_config=AgentRenderConfig(mode="RGBArray") # Set render mode here
        # seed = None # Optional seed
        # normalize_actions = False # Optional override
        # normalize_observations = False # Optional override
    )

    # Aircraft Physics Config (Using builder or defining manually)
    # physics_config = FullAircraftPresetBuilder.get_physics_config(ac_type="twin_otter")
    # Or define specific values if overriding defaults:
    physics_config = AircraftPhysicsConfig(ac_type="twin_otter") # Start with defaults
    # physics_config.max_speed = 250 # Example override

    # Start Config (Using helper)
    trim_cond_dict = FullAircraftPresetBuilder.build_trim_condition(
            trim_type="StraightAndLevel",
            airspeed=80.0
    )
    start_config = build_fixed_start_config(
        position=(0.0, 0.0, -500.0), # Z=0 altitude (AGL if terrain is flat at origin)
        speed=80.0,
        heading_deg=0.0,
        trim_condition=trim_cond_dict
    )

    # Task Config (Control Task for this example)
    task_data = ControlTaskConfigData(
        control_type="Altitude", # Capitalized to match Literal
        target=0.0, # Target ground level
        tolerance=10.0
    )
    task_config = TaskConfig(type="Control", config=task_data)


    # Aircraft Definition
    aircraft_def = AircraftDefinition(
        type="full",
        physics=physics_config,
        start=start_config,
        task=task_config,
        action_type="Continuous",
        observation_type="Continuous",
        normalize_act=False # Explicitly set normalize flags if desired
    )

    # Final Environment Config Object
    final_env_config = SingleAgentEnvConfig(
        env=env_base_config,
        aircraft=aircraft_def
    )

    # --- 2. Create environment using the structured config ---
    logger.info("Creating environment...")
    try:
        # Pass the SingleAgentEnvConfig object
        env = SingleAgentEnv(config=final_env_config, render_mode="rgb_array")
        logger.info("Environment created.")

        # --- 3. Reset environment ---
        logger.info("Resetting environment...")
        # Provide seed here if you want deterministic reset behaviour separate from init seed
        obs, info = env.reset(seed=123)
        logger.info(f"Initial observation shape: {obs.shape}")
        logger.info(f"Initial info: {info}")


        # --- 4. Run simulation ---
        time_points = []
        x_positions = []
        z_positions = []
        velocities = []
        roll_angles = []
        pitch_angles = []

        dt = env.dt # Get dt from the environment instance
        time_elapsed = 0.0
        num_steps = 1000 # Reduced steps for quicker test

        # Record initial state
        time_points.append(time_elapsed)
        x_positions.append(obs[0])
        z_positions.append(-obs[2])
        velocities.append(np.sqrt(obs[6]**2 + obs[7]**2 + obs[8]**2))
        roll_angles.append(np.degrees(obs[3]))
        pitch_angles.append(np.degrees(obs[4]))

        logger.info(f"Starting simulation for {num_steps} steps...")
        # Get the specific trim action calculated by Rust (if available in info)
        # Otherwise, use the known approximate trim action for twin_otter at 80 m/s S&L
        trim_action = info.get("trim_action", {}).get("action")
        if trim_action is None:
             logger.warning("Trim action not found in info, using approximate known value.")
             # Elevator, Aileron, Throttle, Rudder
             trim_action = np.array([-0.0288, 0.0, 0.774, 0.0], dtype=np.float32)
        else:
             # Ensure it's a numpy array with correct shape/type
             trim_action = np.array(list(trim_action.values()), dtype=np.float32)
             logger.info(f"Using trim action from info: {trim_action}")


        for i in range(num_steps):
            # Apply the calculated/known trim action
            action = trim_action

            obs, reward, term, trunc, info = env.step(action)
            time_elapsed += dt # Use env.dt

            # Store data
            time_points.append(time_elapsed)
            x_positions.append(obs[0])
            z_positions.append(-obs[2])
            velocities.append(np.sqrt(obs[6]**2 + obs[7]**2 + obs[8]**2))
            roll_angles.append(np.degrees(obs[3]))
            pitch_angles.append(np.degrees(obs[4]))

            if i % 100 == 0 or i == num_steps - 1:
                logger.info(f"Step {i+1}/{num_steps}, Alt: {z_positions[-1]:.2f}, Spd: {velocities[-1]:.2f}, Pitch: {pitch_angles[-1]:.2f}, Roll: {roll_angles[-1]:.2f}")

            if term or trunc:
                logger.info(f"Episode ended early at step {i}. Term={term}, Trunc={trunc}")
                break

        logger.info("Simulation complete")

    except Exception as e:
         logger.error(f"An error occurred during environment usage: {e}", exc_info=True)
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            logger.info("Closing environment.")
            env.close()

    # --- 5. Plotting (remains the same) ---
    if time_points: # Only plot if simulation ran
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
        # plt.show() # Uncomment to display plot interactively
    else:
        print("No simulation data to plot.")


if __name__ == "__main__":
    main()
