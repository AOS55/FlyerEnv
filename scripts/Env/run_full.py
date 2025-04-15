import gymnasium as gym
# Ensure flyer_env is importable and registers the env
import flyer_env
import time
import numpy as np
import matplotlib.pyplot as plt
import traceback

# --- Registration ---
if hasattr(flyer_env, 'register_flyer_envs'):
     flyer_env.register_flyer_envs()
elif hasattr(flyer_env, 'register_envs'):
     flyer_env.register_envs()
else:
    print("Warning: Could not find standard registration function in flyer_env.")

# --- Imports for Type Checking (Optional) ---
try:
    from flyer_env.envs.common.single_agent_env import SingleAgentEnv
    from flyer_env.envs.common.observation import FullObservation, ObservationType
    from flyer_env.envs.common.action import FullContinuousAction, ActionType
except ImportError as e:
    print(f"Warning: Could not import classes for type checking: {e}")
    SingleAgentEnv, FullObservation, ObservationType = None, None, None
    FullContinuousAction, ActionType = None, None

def run_and_plot(
    initial_zero_steps: int = 75, # Number of steps to apply zero action
    # Define the step input action in normalized [-1, 1] space.
    # Order corresponds to action_handler.features (likely: [elevator, aileron, throttle, rudder])
    step_action_config: list = [0.3, 0.0, 0.6, 0.0] # Example: Apply positive elevator and throttle
):
    """
    Runs the simulation applying zero action for initial_zero_steps,
    then applies a constant step_action_config for the remainder.
    Plots state and control surface data.
    """
    print("--- Starting Full Aircraft Plotting Test with Step Input ---")
    print(f"Applying zero action for first {initial_zero_steps} steps.")
    print(f"Then applying step action: {step_action_config}")

    wrapped_env = None       # The object returned by gym.make (potentially wrapped)
    unwrapped_env = None     # The core environment instance
    obs_handler = None       # The observation handler object
    action_handler = None    # The action handler object

    try:
        # 1. Create the environment
        wrapped_env = gym.make("flyer_control-v1",
            seed=42,
            use_full_aircraft=True,
            control_type="altitude", # Task setup might influence starting state/rewards
            target_value=1000.0,
            tolerance=500.0, # Wider tolerance as we're not controlling to target here
            # Make episode long enough to see effects
            env_config={"max_episode_steps": 300, "steps_per_action": 5}
        )
        print(f"[SUCCESS] Raw environment created. Type: {type(wrapped_env)}")

        # 2. Get the unwrapped environment
        unwrapped_env = wrapped_env.unwrapped
        print(f"Accessed unwrapped environment. Type: {type(unwrapped_env)}")

        # 3. Get Handlers from the UNWRAPPED environment
        if not hasattr(unwrapped_env, 'vehicle'): raise AttributeError("Unwrapped env lacks 'vehicle'")
        obs_handler = unwrapped_env.vehicle.observation
        action_handler = unwrapped_env.vehicle.action
        print(f"Observation Handler accessed. Type: {type(obs_handler)}")
        print(f"Action Handler accessed. Type: {type(action_handler)}")
        if not hasattr(obs_handler, 'to_dict'): raise AttributeError("Obs handler lacks 'to_dict'")
        if not hasattr(action_handler, 'to_dict'): raise AttributeError("Action handler lacks 'to_dict'")
        if hasattr(action_handler, 'features'): print(f"  Action features: {action_handler.features}")

        # --- Define Normalized Actions ---
        action_space = unwrapped_env.action_space
        if not isinstance(action_space, gym.spaces.Box):
             raise TypeError(f"Expected Box action space, got {type(action_space)}")

        # Zero action (all zeros)
        zero_action_norm = np.zeros(action_space.shape, dtype=action_space.dtype)
        # Step action (from config)
        step_action_norm = np.array(step_action_config, dtype=action_space.dtype)

        # Validate step_action fits within action space bounds [-1, 1]
        if not action_space.contains(step_action_norm):
             print(f"Warning: Provided step_action_config {step_action_config} is outside the action space bounds {action_space.low} to {action_space.high}. Clipping.")
             step_action_norm = np.clip(step_action_norm, action_space.low, action_space.high)
             print(f"Using clipped step action: {step_action_norm}")


        # --- Data Storage Initialization ---
        data = { # Use keys consistent with FullObservation features
            'steps': [], 'rewards': [], 'x': [], 'y': [], 'z': [],
            'roll': [], 'pitch': [], 'yaw': [], 'u': [], 'v': [], 'w': [],
            'p': [], 'q': [], 'r': [],
            'elevator': [], 'aileron': [], 'throttle': [], 'rudder': [], # Controls
        }
        state_plot_keys = ['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
        action_plot_keys = ['elevator', 'aileron', 'throttle', 'rudder']

        # --- Run Episode ---
        step_count = 0
        start_time = time.time()
        obs, info = wrapped_env.reset(seed=42) # Use WRAPPED env
        done = False
        print("Running simulation episode with step input...")
        print(f"Initial info dict: {info}")

        # Store initial state (step 0)
        state_dict_init = obs_handler.to_dict(obs) # Use handler from unwrapped_env
        data['steps'].append(step_count)
        data['rewards'].append(0.0)
        for key in state_plot_keys: data[key].append(state_dict_init.get(key, np.nan))
        for key in action_plot_keys: data[key].append(np.nan) # No action at step 0

        while not done:
            # Increment step count *before* deciding action for the current step
            step_count += 1

            # --- Action Generation (Step Input Logic) ---
            if step_count <= initial_zero_steps:
                action_norm = zero_action_norm
                action_source = "Zero Action"
            else:
                if step_count == initial_zero_steps + 1: # Print only on first step action
                     print(f"--- Applying Step Input from step {step_count} ---")
                action_norm = step_action_norm
                action_source = "Step Action"

            # --- Action Processing & Storage ---
            try:
                 # Convert to denormalized dictionary using action handler
                 action_dict = action_handler.to_dict(action_norm)
            except Exception as e_action_todict:
                 print(f"[ERROR] Failed to convert action to dict at step {step_count}: {e_action_todict}")
                 for key in action_plot_keys: data[key].append(np.nan) # Store NaNs
                 action_dict = {key: np.nan for key in action_plot_keys} # Use dict of NaNs for printout
            else:
                 # Store denormalized action values for this step if conversion succeeded
                 for key in action_plot_keys:
                     data[key].append(action_dict.get(key, np.nan))

            # --- Environment Step ---
            try:
                # Step using the WRAPPED env with the NORMALIZED action
                obs, reward, terminated, truncated, info = wrapped_env.step(action_norm)
            except Exception as e_step:
                 print(f"\n[ERROR] Exception during env.step() at step {step_count}: {e_step}")
                 traceback.print_exc(); break
            done = terminated or truncated # Check if episode ended

            # --- Observation Processing and Storage ---
            try:
                # Convert obs using handler from UNWRAPPED env
                state_dict = obs_handler.to_dict(obs)
            except Exception as e_todict:
                 print(f"[ERROR] Failed to convert obs to dict at step {step_count}: {e_todict}")
                 # Append NaNs for state to keep lists aligned
                 data['steps'].append(step_count); data['rewards'].append(reward)
                 for key in state_plot_keys: data[key].append(np.nan)
                 continue # Skip to next loop iteration

            # Store state data (if obs processing succeeded)
            data['steps'].append(step_count)
            data['rewards'].append(reward)
            for key in state_plot_keys: data[key].append(state_dict.get(key, np.nan))

            # Periodic printout
            if step_count % 50 == 0 or done:
                 action_str = ", ".join([f"{k}={v:.2f}" for k,v in action_dict.items() if not np.isnan(v)])
                 print(f"Step: {step_count}, Src: {action_source}, Action: {{{action_str}}}, Rwd: {reward:.3f}, Done: {done}")


        end_time = time.time()
        if step_count > 0: print(f"Episode finished after {step_count} steps in {end_time - start_time:.2f} seconds.")
        else: print("Simulation did not run any steps.")

    # --- Error Handling & Cleanup ---
    except Exception as e:
        print(f"\n[ERROR] An error occurred during environment setup or execution:")
        traceback.print_exc()
    finally:
        if wrapped_env is not None:
             print("\nClosing environment...")
             try:
                 wrapped_env.close()
                 print("[SUCCESS] Environment closed.")
             except Exception as close_err:
                 print(f"[ERROR] Exception during env.close(): {close_err}")

    # --- Plotting ---
    if 'data' not in locals() or not data['steps'] or len(data['steps']) <= 1:
        print("Not enough data collected, skipping plots.")
        return

    print("Generating plots...")
    # Convert angles/rates to degrees (includes control surfaces)
    for key in ['roll', 'pitch', 'yaw', 'p', 'q', 'r', 'elevator', 'aileron', 'rudder']:
        if key in data and len(data[key]) > 0:
             valid_data = np.array(data[key], dtype=float)
             with np.errstate(invalid='ignore'): # Ignore NaN comparison warnings
                 if np.nanmax(np.abs(valid_data)) < 2 * np.pi + 0.1: # Heuristic radian check
                     data[key] = np.where(np.isnan(valid_data), np.nan, np.rad2deg(valid_data))
                 else: data[key] = valid_data
    # Ensure throttle is float array
    if 'throttle' in data and len(data['throttle']) > 0: data['throttle'] = np.array(data['throttle'], dtype=float)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Plotting Figures 1-5 (Position, Orientation, Ang Vel, Lin Vel, Reward)
    # (Plotting code remains the same as previous version)
    # Figure 1: Position
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig1.suptitle('Aircraft Position', fontsize=16)
    axs1[0].plot(data['steps'], data['z'], label='Altitude (z)', color='navy')
    axs1[0].set_xlabel('Step Count'); axs1[0].set_ylabel('Altitude (m)'); axs1[0].legend(); axs1[0].grid(True); axs1[0].set_title('Altitude vs Time')
    axs1[1].plot(data['x'], data['y'], label='Trajectory', color='darkorange', marker='.', markersize=2, linestyle='-')
    axs1[1].set_xlabel('X Position (m)'); axs1[1].set_ylabel('Y Position (m)'); axs1[1].axis('equal'); axs1[1].grid(True); axs1[1].set_title('Top-Down Trajectory (X-Y)')
    valid_x = [x for x in data['x'] if not np.isnan(x)]; valid_y = [y for y in data['y'] if not np.isnan(y)]
    if len(valid_x) > 0 and len(valid_y) > 0:
        axs1[1].plot(valid_x[0], valid_y[0], 'go', markersize=8, label='Start')
        axs1[1].plot(valid_x[-1], valid_y[-1], 'rx', markersize=8, label='End')
    axs1[1].legend()

    # Figure 2: Orientation
    fig2, axs2 = plt.subplots(1, 1, figsize=(12, 5))
    fig2.suptitle('Aircraft Orientation (Euler Angles)', fontsize=16)
    axs2.plot(data['steps'], data['roll'], label='Roll', color='tab:blue')
    axs2.plot(data['steps'], data['pitch'], label='Pitch', color='tab:red')
    axs2.plot(data['steps'], data['yaw'], label='Yaw', color='tab:green')
    axs2.set_xlabel('Step Count'); axs2.set_ylabel('Angle (degrees)'); axs2.legend(); axs2.grid(True)

    # Figure 3: Angular Velocities
    fig3, axs3 = plt.subplots(1, 1, figsize=(12, 5), sharex=True)
    fig3.suptitle('Aircraft Angular Velocities (Body Frame)', fontsize=16)
    axs3.plot(data['steps'], data['p'], label='Roll Rate (p)', color='tab:blue')
    axs3.plot(data['steps'], data['q'], label='Pitch Rate (q)', color='tab:red')
    axs3.plot(data['steps'], data['r'], label='Yaw Rate (r)', color='tab:green')
    axs3.set_xlabel('Step Count'); axs3.set_ylabel('Rate (degrees/sec)'); axs3.legend(); axs3.grid(True)

    # Figure 4: Linear Velocities
    fig4, axs4 = plt.subplots(1, 1, figsize=(12, 5), sharex=True)
    fig4.suptitle('Aircraft Linear Velocities (Body Frame)', fontsize=16)
    axs4.plot(data['steps'], data['u'], label='u (Forward)', color='tab:blue')
    axs4.plot(data['steps'], data['v'], label='v (Right)', color='tab:red')
    axs4.plot(data['steps'], data['w'], label='w (Down)', color='tab:green')
    axs4.set_xlabel('Step Count'); axs4.set_ylabel('Velocity (m/s)'); axs4.legend(); axs4.grid(True)

    # Figure 5: Reward
    fig5, axs5 = plt.subplots(1, 1, figsize=(12, 4))
    fig5.suptitle('Episode Reward', fontsize=16)
    if len(data['steps']) > 1:
        axs5.plot(data['steps'][1:], data['rewards'][1:], label='Reward per Step', color='purple', marker='.', linestyle='-')
        axs5.set_xlabel('Step Count'); axs5.set_ylabel('Reward'); axs5.legend(); axs5.grid(True)

    # Figure 6: Control Surfaces
    fig6, axs6 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig6.suptitle('Control Surface Inputs (Denormalized)', fontsize=16)
    # Skip step 0 (NaN action) using [1:] slicing
    axs6[0].plot(data['steps'][1:], data['elevator'][1:], label='Elevator', color='tab:purple', drawstyle='steps-post')
    axs6[0].plot(data['steps'][1:], data['aileron'][1:], label='Aileron', color='tab:brown', drawstyle='steps-post')
    axs6[0].plot(data['steps'][1:], data['rudder'][1:], label='Rudder', color='tab:pink', drawstyle='steps-post')
    axs6[0].set_ylabel('Angle (degrees)')
    axs6[0].legend()
    axs6[0].grid(True)
    axs6[0].set_title('Surface Angles')
    # Add vertical line where step input begins
    axs6[0].axvline(initial_zero_steps + 0.5, color='gray', linestyle='--', label=f'Step Input Start (t={initial_zero_steps+1})')
    axs6[0].legend()


    axs6[1].plot(data['steps'][1:], data['throttle'][1:], label='Throttle', color='tab:gray', drawstyle='steps-post')
    axs6[1].set_xlabel('Step Count')
    axs6[1].set_ylabel('Throttle Setting (0-1)')
    axs6[1].set_ylim(-0.1, 1.1)
    axs6[1].grid(True)
    axs6[1].set_title('Throttle')
    # Add vertical line where step input begins
    axs6[1].axvline(initial_zero_steps + 0.5, color='gray', linestyle='--', label=f'Step Input Start (t={initial_zero_steps+1})')
    axs6[1].legend()


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Example: Apply zero action for 75 steps, then 30% elevator up and 60% throttle
    run_and_plot(initial_zero_steps=75, step_action_config=[0.03, 0.0, 0.6, 0.0])

    # Example: Apply zero action for 50 steps, then 20% right aileron
    # run_and_plot(initial_zero_steps=50, step_action_config=[0.0, 0.2, 0.5, 0.0]) # Assuming throttle needed too

    # Example: Apply zero action for 100 steps, then 30% right rudder
    # run_and_plot(initial_zero_steps=100, step_action_config=[0.0, 0.0, 0.5, 0.3]) # Assuming throttle needed too