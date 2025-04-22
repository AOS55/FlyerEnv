import os
import csv
import math
import matplotlib.pyplot as plt
import flyer_env # Assuming this registers the env
import gymnasium as gym
import numpy as np
import scipy.linalg
from typing import Dict, List, Tuple

# ==============================================================================
# --- Configuration & Constants ---
# ==============================================================================

# --- Simulation Settings ---
TIME_STEP_LIMIT = 10000      # Simulation duration (steps)
SIM_LOG_INTERVAL = 100       # Print status every N steps (increased from 50)
OUTPUT_DIR = "run_output_lqr_full_aircraft" # Directory for results
SAVE_RESULTS = True          # Save history to CSV?

# --- Model & Trim ---
# <<< CHOOSE THE TARGET TRIM AIRSPEED HERE >>>
# This airspeed MUST match the conditions for which A_full/B_full/X_TRIM_FULL were derived.
TARGET_TRIM_AIRSPEED = 80.0  # m/s - Example value

# Use the chosen target airspeed for linearization assumption
LINEARIZATION_AIRSPEED = TARGET_TRIM_AIRSPEED
print(f"INFO: Setting Linearization Airspeed Assumption to: {LINEARIZATION_AIRSPEED} m/s")

# !!! IMPORTANT !!!
# The A_full, B_full matrices AND X_TRIM_FULL below MUST correspond
# to the TARGET_TRIM_AIRSPEED chosen above for the Twin Otter model used in flyer_env.
# The current hardcoded values ARE LIKELY INCORRECT for 80.0 m/s.
# You NEED to obtain the correct A/B matrices and the corresponding
# full 12-element state trim vector [u, v, w, p, q, r, phi, theta, psi, x, y, z]
# for TARGET_TRIM_AIRSPEED and straight, level flight.

# Order of states in X_TRIM_FULL must match ORIGINAL_STATE_ORDER
ORIGINAL_STATE_ORDER =  ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']
N_STATES_FULL = len(ORIGINAL_STATE_ORDER) # Should be 12

# Define the FULL trim state vector corresponding to the linearization point
# These are the target values the LQR will try to hold.
# Placeholder values - YOU MUST PROVIDE THE CORRECT VALUES FOR YOUR MODEL/SPEED
X_TRIM_FULL = np.array([
    TARGET_TRIM_AIRSPEED,  # u_trim (m/s) - SET TO TARGET AIRSPEED
    0.0,                   # v_trim (m/s) - Assumed 0 for straight flight
    1.564,                 # w_trim (m/s) - NEEDS UPDATE FOR 80 m/s
    0.0,                   # p_trim (rad/s) - Assumed 0 for level flight
    -4.826e-2,             # q_trim (rad/s) - NEEDS UPDATE FOR 80 m/s
    0.0,                   # r_trim (rad/s) - Assumed 0 for straight flight
    0.0,                   # phi_trim (rad) - Assumed 0 for level flight
    0.020,                 # theta_trim (rad) - NEEDS UPDATE FOR 80 m/s
    0.0,                   # psi_trim (rad) - Assumed 0 for initial heading reference
    0.0,                   # pos_x_trim (m) - Assumed 0 (LQR focuses on deviations)
    0.0,                   # pos_y_trim (m) - Assumed 0
    0.0                    # pos_z_trim (m) - Assumed 0 (Altitude is implicitly handled by u,w,theta control)
])
print(f"WARNING: Using Placeholder X_TRIM_FULL = {X_TRIM_FULL}.")
print(f"Ensure ALL components are correct for straight, level flight at {TARGET_TRIM_AIRSPEED} m/s!")
assert len(X_TRIM_FULL) == N_STATES_FULL, f"X_TRIM_FULL length mismatch. Expected {N_STATES_FULL}"


# Order of inputs for U_TRIM_FULL must match ORIGINAL_INPUT_ORDER
ORIGINAL_INPUT_ORDER = ['elevator', 'aileron', 'rudder', 'throttle']
N_INPUTS_FULL = len(ORIGINAL_INPUT_ORDER) # Should be 4

# Define the FULL trim input vector corresponding to X_TRIM_FULL
# YOU MUST PROVIDE THE CORRECT VALUES FOR YOUR MODEL/SPEED
# (Elevator/Throttle likely non-zero, Aileron/Rudder are specified as 0 by user)
U_TRIM_FULL = np.array([
    -0.028833359,          # elevator_trim (rad) - NEEDS UPDATE FOR 80 m/s
    0.0,                   # aileron_trim (rad) - Specified as 0
    0.0,                   # rudder_trim (rad) - Specified as 0
    0.77409241             # throttle_trim (-) - NEEDS UPDATE FOR 80 m/s
])
print(f"WARNING: Using Placeholder U_TRIM_FULL = {U_TRIM_FULL}.")
print(f"Ensure elevator/throttle components are correct for trim at {TARGET_TRIM_AIRSPEED} m/s!")
assert len(U_TRIM_FULL) == N_INPUTS_FULL, f"U_TRIM_FULL length mismatch. Expected {N_INPUTS_FULL}"

# --- LQR Tuning ---
# Penalties for state deviations [u, v, w, p, q, r, phi, theta, psi, x, y, z]
# *** THESE ARE EXAMPLES - TUNING IS REQUIRED ***
Q_FULL_DIAG = [
    1.0,    # u (forward speed)
    0.1,    # v (side speed)
    0.0,    # w (vertical speed - body frame)
    0.01,   # p (roll rate)
    0.01,   # q (pitch rate)
    0.01,    # r (yaw rate)
    0.01,  # phi (roll angle)
    0.01,   # theta (pitch angle)
    0.01,    # psi (yaw angle) - Lower penalty if heading hold isn't critical
    0.0,    # pos_x - Zero penalty usually best for stabilization LQR
    0.0,    # pos_y - Zero penalty
    0.0     # pos_z - Zero penalty (Altitude stabilized via w/theta)
]
Q_SCALING = 1.0
# Penalties for control effort [elevator, aileron, rudder, throttle]
# *** THESE ARE EXAMPLES - TUNING IS REQUIRED ***
R_FULL_DIAG = [
    10.0,   # elevator
    1000.0,   # aileron
    1000.0,   # rudder - Often higher penalty if minimizing sideslip is desired indirectly
    1.0     # throttle
]
R_SCALING = 5.0

# Order expected/provided by the Gymnasium environment `flyer_env`
# !!! VERIFY THIS MATCHES YOUR `flyer_env` !!!
ENV_OBS_ORDER = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'u', 'v', 'w', 'p', 'q', 'r']
ENV_ACTION_ORDER = ['elevator', 'aileron', 'throttle', 'rudder'] # Note: Order differs from ORIGINAL_INPUT_ORDER

# --- Environment Settings ---
ENV_ID = "flyer_control-v1" # Make sure this matches your registered env name
ENV_SEED = 42
ENV_RENDER_MODE = "rgb_array" # Use None for faster runs
USE_FULL_AIRCRAFT = True # Must be True for FullAircraftPreset/trim features


# ==============================================================================
# --- Original Full System Matrices (Reference) ---
# ==============================================================================
print(f"WARNING: Using hardcoded A_full/B_full matrices. Ensure these are correct for {TARGET_TRIM_AIRSPEED} m/s!")
# State order: u, v, w, p, q, r, phi, theta, psi, pos_x, pos_y, pos_z (12 states)
# Input order: elevator, aileron, rudder, throttle (4 inputs)
A_full = np.array([
    [-8.24385309e-2, 0.00000000e0, -5.86910992e-2, 0.00000000e0, -1.12947400e0, 0.00000000e0, -1.38777878e-10, -9.80811045e0, -2.49800181e-10, 0.00000000e0, 0.00000000e0, -1.44742579e-4],
    [0.00000000e0, -3.29346884e-1, 0.00000000e0, 1.23221692e0, 0.00000000e0, -7.34131501e1, 9.80811055e0, 0.00000000e0, 3.97046694e-15, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [-2.28565094e-1, 0.00000000e0, -1.82447944e0, 0.00000000e0, 7.00546464e1, 0.00000000e0, -4.90318897e-6, -1.92534101e-1, 0.00000000e0, 0.00000000e0, 0.00000000e0, -8.13516809e-4],
    [0.00000000e0, -1.39042191e-1, 0.00000000e0, -5.20063842e0, 0.00000000e0, 2.28826981e0, 5.88866587e-17, 0.00000000e0, 1.88437308e-15, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [9.32849464e-4, 0.00000000e0, -1.60801699e-1, -4.21329638e-8, -2.84217853e0, 4.21329638e-8, 1.38777878e-10, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 6.21874774e-6],
    [0.00000000e0, 5.69790487e-2, 0.00000000e0, -4.27540816e-1, 0.00000000e0, -2.84569710e0, -2.41315645e-17, 0.00000000e0, -7.72210065e-16, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 1.00000000e0, 0.00000000e0, 1.96295909e-2, -9.47365395e-4, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 1.00000000e0, 0.00000000e0, 2.41334730e-8, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 1.00019264e0, -4.82714033e-2, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [9.99807398e-1, 0.00000000e0, 1.96258156e-2, 0.00000000e0, 0.00000000e0, 0.00000000e0, -1.42108547e-8, -3.98330258e-5, -3.98330258e-5, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 1.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0, -1.56379667e0, 0.00000000e0, 7.96806173e1, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [-1.96258101e-2, 0.00000000e0, 9.99807395e-1, 0.00000000e0, 0.00000000e0, 0.00000000e0, -7.81819054e-7, -7.96806173e1, 0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0]
])
B_full = np.array([
    [-6.63832891e-2, 0.00000000e0, 0.00000000e0, 4.45360444e0],
    [0.00000000e0, -1.51228329e0, -5.72295443e0, 0.00000000e0],
    [-8.62891056e0, 0.00000000e0, 0.00000000e0, -4.44089210e-10],
    [0.00000000e0, 2.08736312e1, 1.14857665e1, 0.00000000e0],
    [1.41907252e1, 0.00000000e0, 0.00000000e0, 1.98251878e-1],
    [0.00000000e0, 1.81820734e0, -4.47437870e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0],
    [0.00000000e0, 0.00000000e0, 0.00000000e0, 0.00000000e0]
])
assert A_full.shape == (N_STATES_FULL, N_STATES_FULL), "A_full shape mismatch"
assert B_full.shape == (N_STATES_FULL, N_INPUTS_FULL), "B_full shape mismatch"


# --- Define Actuator Limits ---
# (Used later for clipping the *total* command)
ACTION_LIMITS = {
    'elevator': (math.radians(-25), math.radians(25)),
    'aileron': (math.radians(-20), math.radians(20)), # Example limits
    'rudder': (math.radians(-30), math.radians(30)),   # Example limits
    'throttle': (0.0, 1.0)
}

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
# No extract_longitudinal_model needed anymore

def calculate_lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solves the Continuous Algebraic Riccati Equation (CARE) and computes the LQR gain K.
    """
    print("\n--- Calculating LQR Gain K ---")
    n_states = A.shape[0]
    n_inputs = B.shape[1]
    print(f"System Dimensions: States={n_states}, Inputs={n_inputs}")
    print(f"Q matrix ({Q.shape}):\n{Q}")
    print(f"R matrix ({R.shape}):\n{R}")

    # --- Solve CARE ---
    try:
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        print(f"\nSolved CARE, P matrix ({P.shape}):\n{P}")
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"\nERROR solving CARE: {e}")
        print("Check if the system (A, B) is controllable and (A, sqrt(Q)) is observable.")
        # Optional: Add controllability check here if needed
        # from control.matlab import ctrb
        # Wc = ctrb(A, B)
        # print(f"Controllability Matrix Rank: {np.linalg.matrix_rank(Wc)} (Expected: {n_states})")
        raise SystemExit("Cannot proceed without solving CARE.") from e

    # --- Calculate Gain K = R^-1 * B^T * P ---
    try:
        R_inv = np.linalg.inv(R)
        # print(f"\nInverse of R ({R_inv.shape}):\n{R_inv}") # Less verbose
    except np.linalg.LinAlgError:
        print("ERROR: R matrix is singular, cannot compute inverse.")
        raise SystemExit("Cannot calculate LQR gain with singular R matrix.")

    B_transpose = B.T
    # print(f"\nTranspose of B ({B_transpose.shape}):\n{B_transpose}") # Less verbose

    # K = R_inv @ B.T @ P
    K = R_inv @ B_transpose @ P
    print(f"\nCalculated LQR Gain Matrix K ({K.shape}):\n{K}")

    # --- Stability Check (Optional but Recommended) ---
    A_cl = A - B @ K # Closed-loop system matrix
    try:
        eigenvalues_cl = np.linalg.eigvals(A_cl)
        print("\nClosed-Loop Eigenvalues (A - B*K):")
        for eig in eigenvalues_cl:
            print(f"  {eig.real:.4f} + {eig.imag:.4f}j")
        if np.all(np.real(eigenvalues_cl) < 0):
            print("--> Closed-loop system is STABLE.")
        else:
            print("--> WARNING: Closed-loop system may be UNSTABLE (eigenvalues with non-negative real parts found).")
    except np.linalg.LinAlgError:
        print("\nWARNING: Could not compute eigenvalues for stability check.")

    return K

def get_env_indices(env_list: List[str], target_list: List[str]) -> List[int]:
    """ Finds indices of target items within the environment list. """
    try:
        indices = [env_list.index(name) for name in target_list]
        return indices
    except ValueError as e:
        # Find which item caused the error
        missing_item = None
        for name in target_list:
            if name not in env_list:
                missing_item = name
                break
        print(f"ERROR: Item '{missing_item}' in target list {target_list} not found in environment list '{env_list}'.")
        raise

# ==============================================================================
# --- Full 6-DOF LQR Controller Class ---
# ==============================================================================
class FullLQRController:
    """
    Calculates full control action *deviations* (elevator, aileron, rudder, throttle)
    using a pre-computed full-order LQR gain matrix K_full.
    Requires the full observation vector, full trim state vector,
    and indices mapping env observation to the LQR state order.
    """
    def __init__(self, K_full: np.ndarray, full_state_indices_in_env: List[int], full_input_order: List[str]):
        self.K_full = K_full
        self.n_states_full = K_full.shape[1] # Should be 12
        self.n_inputs_full = K_full.shape[0] # Should be 4
        self.full_state_indices_in_env = full_state_indices_in_env
        self.full_input_order = full_input_order # e.g., ['elevator', 'aileron', 'rudder', 'throttle']

        assert len(full_state_indices_in_env) == self.n_states_full, "Mismatch between K_full states and provided indices"
        assert len(full_input_order) == self.n_inputs_full, "Mismatch between K_full inputs and input order list"
        assert self.K_full.shape == (self.n_inputs_full, self.n_states_full), f"K_full shape mismatch, expected ({self.n_inputs_full}, {self.n_states_full}), got {self.K_full.shape}"


        print("\n--- Full LQR Controller Initialized ---")
        print(f"Using K_full gain matrix ({self.K_full.shape})")
        print(f"Extracting states at indices {self.full_state_indices_in_env} from environment observation.")
        print(f"Expecting trim state vector of size {self.n_states_full}.")
        print(f"Outputting actions deviations for: {self.full_input_order}")
        # Note: Clipping of the *total* action happens in the main loop


    def compute_action_deviation(self, obs_env_order: np.ndarray, x_trim_full: np.ndarray) -> Dict[str, float]:
        """
        Calculates the raw control deviations (u_dev = -K*x_dev) based on
        current observation and full trim state. Returns deviations as a dict.
        """
        # --- Input Validation ---
        # Ensure env observation has enough elements based on the maximum index needed
        max_req_idx = max(self.full_state_indices_in_env) if self.full_state_indices_in_env else -1
        if len(obs_env_order) <= max_req_idx:
             raise ValueError(f"Observation vector size ({len(obs_env_order)}) is too small. Need at least {max_req_idx + 1} elements based on indices {self.full_state_indices_in_env}.")
        if len(x_trim_full) != self.n_states_full:
             raise ValueError(f"Full trim state size ({len(x_trim_full)}) mismatch. Expected {self.n_states_full}.")

        # --- State Extraction ---
        # Get the current values of the full state vector from the environment observation
        # The order here matches LQR state order (e.g., ORIGINAL_STATE_ORDER)
        x_full_current = obs_env_order[self.full_state_indices_in_env]
        # print(f"  [Ctrl] Extracted x_full_current: {np.round(x_full_current, 3)}") # DEBUG

        # --- Calculate State Deviation ---
        # Deviation = Current State - Trim State
        # TODO: Consider angle wrapping for phi, theta, psi if large deviations occur
        x_deviation_full = x_full_current - x_trim_full
        # print(f"  [Ctrl] Trim state x_trim_full:      {np.round(x_trim_full, 3)}") # DEBUG
        # print(f"  [Ctrl] State Deviation x_dev_full: {np.round(x_deviation_full, 3)}") # DEBUG

        # --- LQR Control Law ---
        # Control Action Deviation = -K * Deviation
        # u_deviation_raw is shape (4,) -> [ele_dev, ail_dev, rud_dev, thr_dev] (order matches full_input_order)
        u_deviation_raw = -self.K_full @ x_deviation_full
        # print(f"  [Ctrl] Raw deviation u_dev_raw: {np.round(u_deviation_raw, 4)}") # DEBUG

        # --- Return Deviations as Dictionary ---
        # Clipping of the *total* action (trim + deviation) will happen in the main loop
        action_dev_dict = {}
        for i, input_name in enumerate(self.full_input_order):
            action_dev_dict[input_name] = u_deviation_raw[i]

        # print(f"  [Ctrl] Final action_dev_dict: {action_dev_dict}") # DEBUG
        return action_dev_dict


# ==============================================================================
# --- Plotting and History Functions (Modified for Full Aircraft) ---
# ==============================================================================
def plot_full_results(history: Dict[str, list], full_state_plot_order: List[str], full_input_order: List[str]):
    """ Plot key state deviations and all inputs. """
    if not history or not history.get('time'):
        print("History is empty, cannot plot.")
        return

    time = history['time']
    if not time: # Check if time list itself is empty
        print("History has no time entries, cannot plot.")
        return


    print("\n--- Plotting Results ---")

    # Define labels for plots
    state_labels = {
        'u': 'u (m/s)', 'v': 'v (m/s)', 'w': 'w (m/s)',
        'p': 'p (rad/s)', 'q': 'q (rad/s)', 'r': 'r (rad/s)',
        'phi': 'phi (rad)', 'theta': 'theta (rad)', 'psi': 'psi (rad)',
        'pos_x': 'x (m)', 'pos_y': 'y (m)', 'pos_z': 'z (m)' # Deviations often less meaningful
    }
    input_labels = {
        'elevator': 'elevator (rad)', 'aileron': 'aileron (rad)',
        'rudder': 'rudder (rad)', 'throttle': 'throttle'
    }
    # Use a subset of states for plotting deviations to avoid clutter
    plot_state_dev_keys = full_state_plot_order

    # --- Plot State Deviations ---
    n_states_plot = len(plot_state_dev_keys)
    n_rows = math.ceil(n_states_plot / 3) # Aim for 3 plots per row
    plt.figure(figsize=(18, 5 * n_rows))
    plt.suptitle('Full LQR Control: Key State Deviation Evolution (Target = 0)', fontsize=16)
    for i, key in enumerate(plot_state_dev_keys):
        plt.subplot(n_rows, 3, i + 1)
        # Look for the deviation key (e.g., 'u_dev')
        dev_key = f"{key}_dev"
        if dev_key not in history or not history[dev_key]:
            print(f"Warning: State deviation '{dev_key}' not found or empty in history.")
            plt.title(f'Deviation: {state_labels.get(key, key)} (No Data)')
            continue

        actual_deviation = np.array(history[dev_key])
        label_rad = state_labels.get(key, key)

        # Plot in degrees if it's an angle/rate typically viewed in degrees
        if key in ['p', 'q', 'r', 'phi', 'theta', 'psi']:
            label_deg = label_rad.replace("rad", "deg")
            plt.plot(time, np.degrees(actual_deviation), label='Actual Deviation')
            plt.ylabel(label_deg)
            plt.title(f'Deviation: {label_deg}')
        else:
            plt.plot(time, actual_deviation, label='Actual Deviation')
            plt.ylabel(label_rad)
            plt.title(f'Deviation: {label_rad}')

        plt.axhline(0.0, color='r', linestyle='--', label='Target (Zero Deviation)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap

    # --- Plot Control Inputs (Total Applied) ---
    n_inputs = len(full_input_order)
    plt.figure(figsize=(6 * n_inputs, 5))
    plt.suptitle('Full LQR Control: Control Inputs (Total Applied)', fontsize=16)
    for i, key in enumerate(full_input_order):
        plt.subplot(1, n_inputs, i + 1)
        # Look for the total applied action key (e.g., 'elevator_total')
        total_key = f"{key}_total"
        if total_key not in history or not history[total_key]:
            print(f"Warning: Total input '{total_key}' not found or empty in history.")
            plt.title(f'Input: {input_labels.get(key, key)} (No Data)')
            continue

        actual_input = np.array(history[total_key])
        label_rad = input_labels.get(key, key)

        if key in ['elevator', 'aileron', 'rudder']: # Plot angles in degrees
             label_deg = label_rad.replace("rad", "deg")
             plt.plot(time, np.degrees(actual_input))
             plt.ylabel(label_deg)
             plt.title(f'Input: {label_deg}')
        else: # Plot others (throttle) as is
             plt.plot(time, actual_input)
             plt.ylabel(label_rad)
             plt.title(f'Input: {label_rad}')

        # Optional: Add lines for trim values if helpful
        trim_val = U_TRIM_FULL[ORIGINAL_INPUT_ORDER.index(key)]
        if key in ['elevator', 'aileron', 'rudder']:
            plt.axhline(np.degrees(trim_val), color='g', linestyle=':', label=f'Trim ({np.degrees(trim_val):.2f} deg)')
        else:
             plt.axhline(trim_val, color='g', linestyle=':', label=f'Trim ({trim_val:.3f})')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def update_history(history: Dict[str, list], current_time: float,
                   state_deviation_full: np.ndarray, full_state_order: List[str],
                   action_total_dict: Dict[str, float], full_input_order: List[str]):
    """Appends current step's full aircraft data to the history dictionary."""
    history['time'].append(current_time)

    # Log state deviations (using "_dev" suffix)
    for i, key in enumerate(full_state_order):
        history.setdefault(f"{key}_dev", []).append(state_deviation_full[i])

    # Log total actions applied (using "_total" suffix)
    for key in full_input_order:
        print(f"key: {key}")
        # Use .get to handle potential missing keys gracefully
        history.setdefault(f"{key}_total", []).append(action_total_dict.get(f"{key}_total", np.nan))


def save_history_to_csv(history: Dict[str, list], output_dir: str, filename: str):
    """Saves the history dictionary to a CSV file."""
    if not output_dir:
        print("Output directory not set. Skipping saving results.")
        return
    if not history or not history.get('time'):
        print("History is empty, cannot save CSV.")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    print(f"\n--- Saving Results to CSV ---")
    print(f"File path: {file_path}")

    try:
        # Ensure all lists have the same length for zipping
        num_entries = len(history['time'])
        headers = list(history.keys())
        data_columns = []
        for header in headers:
            column = history.get(header, [])
            # Pad with NaN if a column is unexpectedly shorter
            if len(column) < num_entries:
                 print(f"Warning: History column '{header}' has length {len(column)}, expected {num_entries}. Padding with NaN.")
                 column.extend([np.nan] * (num_entries - len(column)))
            data_columns.append(column[:num_entries]) # Ensure consistent length

        # Transpose data for writerows (each row is a time step)
        rows = zip(*data_columns)

        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers) # Write header row
            writer.writerows(rows)   # Write data rows
        print(f"Results successfully saved.")

    except Exception as e:
        print(f"ERROR saving results to CSV: {e}")

# ==============================================================================
# --- Main Simulation Function ---
# ==============================================================================

def run_simulation():
    """
    Sets up the Full 6-DOF LQR controller and runs the simulation loop.
    """
    print("="*60)
    print("--- Starting Full Aircraft LQR Control Simulation ---")
    print("="*60)
    print(f"Using Linearization Airspeed Assumption: {LINEARIZATION_AIRSPEED} m/s")
    print(f"Target Full Trim State (Order: {ORIGINAL_STATE_ORDER}):\n {np.round(X_TRIM_FULL, 4)}")
    print(f"Target Full Trim Input (Order: {ORIGINAL_INPUT_ORDER}):\n {np.round(U_TRIM_FULL, 4)}")
    print(f"Attempting to start simulation at speed: {TARGET_TRIM_AIRSPEED} m/s")

    # --- 1. System Matrices (Using Full System) ---
    # A_full, B_full are already defined globally
    print("\n--- Using Full System Matrices ---")
    print(f"A_full shape: {A_full.shape}, B_full shape: {B_full.shape}")


    # --- 2. Design LQR Controller ---
    print("\n--- Designing Full LQR Controller ---")
    assert len(Q_FULL_DIAG) == N_STATES_FULL, f"Q_FULL_DIAG length mismatch ({len(Q_FULL_DIAG)} vs {N_STATES_FULL})"
    assert len(R_FULL_DIAG) == N_INPUTS_FULL, f"R_FULL_DIAG length mismatch ({len(R_FULL_DIAG)} vs {N_INPUTS_FULL})"

    Q_full = np.diag(Q_FULL_DIAG) * Q_SCALING
    R_full = np.diag(R_FULL_DIAG) * R_SCALING
    K_full = calculate_lqr_gain(A_full, B_full, Q_full, R_full)

    # --- 3. Setup Environment ---
    print("\n--- Setting up Environment ---")
    print(f"Env ID: {ENV_ID}, Seed: {ENV_SEED}, Render: {ENV_RENDER_MODE}")

    # Define the trim condition dictionary to pass to the environment
    env_trim_condition = {
        "type": "StraightAndLevel",
        "airspeed": TARGET_TRIM_AIRSPEED # Use the consistent airspeed
    }
    print(f"Configuring environment with trim condition: {env_trim_condition}")
    print(f"Using ControlFlyerEnv args to set start speed near {TARGET_TRIM_AIRSPEED} m/s")

    try:
        # --- Use ControlFlyerEnv arguments directly ---
        env = gym.make(ENV_ID,
                       seed=ENV_SEED,
                       render_mode=ENV_RENDER_MODE,
                       # --- Key arguments for setting start speed & trim ---
                       control_type="speed",            # Use speed control logic for start setup
                       target_value=TARGET_TRIM_AIRSPEED, # Set the target start speed
                       start_deviation=0.0,           # Set zero deviation for specific start speed
                       trim_condition_params=env_trim_condition,   # Configure aircraft model for trim
                       # --- Other necessary arguments ---
                       use_full_aircraft=True,        # Use the full aircraft model (required for trim)
                       max_episode_steps=TIME_STEP_LIMIT,   # Sync episode length
                       tolerance=1.0,                 # Tolerance for speed task (less relevant for LQR)
                       )

        # Get simulation timestep
        try:
            dt = env.unwrapped.dt
            print(f"Environment dt: {dt:.6f} s")
        except AttributeError:
            dt_fallback = 1/60.0 # Default guess
            try: # Check config if possible
                dt_fallback = env.unwrapped.config.get("time_step", dt_fallback)
            except AttributeError: pass
            dt = dt_fallback
            print(f"Warning: Cannot access env.unwrapped.dt, assuming dt={dt:.6f} s")

    except Exception as e:
        print(f"ERROR creating environment '{ENV_ID}': {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit("Environment creation failed.")

    # --- 4. Get Mappings for Environment ---
    print("\n--- Establishing Environment Mappings ---")
    print(f"LQR State Order (Original): {ORIGINAL_STATE_ORDER}")
    print(f"LQR Input Order (Original): {ORIGINAL_INPUT_ORDER}")
    print(f"Environment Observation Order: {ENV_OBS_ORDER}")
    print(f"Environment Action Order:      {ENV_ACTION_ORDER}")

    # Indices needed by the controller: Indices of LQR states in the *environment observation* vector
    try:
        env_obs_dict = {name: i for i, name in enumerate(ENV_OBS_ORDER)}
        env_act_dict = {name: i for i, name in enumerate(ENV_ACTION_ORDER)}
        # Map ORIGINAL_STATE_ORDER to the indices in ENV_OBS_ORDER
        full_state_indices_in_env = get_env_indices(ENV_OBS_ORDER, ORIGINAL_STATE_ORDER)
        print(f"Indices of LQR States ({len(ORIGINAL_STATE_ORDER)}) in Env Obs ({len(ENV_OBS_ORDER)}): {full_state_indices_in_env}")
    except (ValueError, KeyError, IndexError) as e:
        print(f"ERROR: Failed to map state/action names to environment indices: {e}")
        print("Ensure ORIGINAL_STATE_ORDER/ORIGINAL_INPUT_ORDER and ENV_OBS_ORDER/ENV_ACTION_ORDER are correct and consistent.")
        env.close()
        raise SystemExit("Index mapping failed.")

    # --- 5. Instantiate Controller ---
    # Note: Pass ORIGINAL_INPUT_ORDER to controller as K_full maps state devs to this order
    controller = FullLQRController(K_full, full_state_indices_in_env, ORIGINAL_INPUT_ORDER)

    # --- 6. Initialize Simulation (Reset) ---
    print("\n--- Initializing Simulation ---")
    try:
        # Reset now uses the start configuration implicitly set by __init__ args
        obs_env_order_initial, info = env.reset(seed=ENV_SEED) # Pass seed again if needed
        print(f"Initial Observation (Env Order, len={len(obs_env_order_initial)}):")
        print(f"  {np.round(obs_env_order_initial, 3)}")

        # <<< Verify Initial Speed >>>
        initial_u_speed = obs_env_order_initial[env_obs_dict['u']]
        print(f"  Initial u speed: {initial_u_speed:.3f} m/s (Target was {TARGET_TRIM_AIRSPEED})")
        if not np.isclose(initial_u_speed, TARGET_TRIM_AIRSPEED, atol=1.0): # Check within reasonable tolerance
             print(f"  WARNING: Initial speed ({initial_u_speed:.1f}) not close to target ({TARGET_TRIM_AIRSPEED:.1f}). Start config might not be precise.")

    except Exception as e:
        print(f"ERROR during env.reset(): {e}")
        env.close()
        raise SystemExit("Environment reset failed.")

    # --- Verify observation length ---
    if len(obs_env_order_initial) != len(ENV_OBS_ORDER):
        print(f"ERROR: Initial observation length ({len(obs_env_order_initial)}) doesn't match expected ({len(ENV_OBS_ORDER)} based on ENV_OBS_ORDER).")
        env.close()
        raise SystemExit("Observation length mismatch.")

    current_obs_env = obs_env_order_initial.copy()

    # Initialize history dictionary
    # Log state deviations (_dev suffix) and total applied actions (_total suffix)
    history_keys = ['time']
    history_keys += [f"{s}_dev" for s in ORIGINAL_STATE_ORDER]
    history_keys += [f"{i}_total" for i in ORIGINAL_INPUT_ORDER]
    history = {key: [] for key in history_keys}
    print(f"Initialized history log for {len(history_keys)} keys.")

    print(f"\nStarting simulation from environment's reset state (attempted start speed {TARGET_TRIM_AIRSPEED} m/s).")
    print(f"LQR controller will act on deviations from full trim: {np.round(X_TRIM_FULL, 4)}")


    print("\n" + "="*60)
    print(f"--- Starting Simulation Loop ({TIME_STEP_LIMIT} steps) ---")
    print("="*60)

    # --- 7. Simulation Loop ---
    for step in range(TIME_STEP_LIMIT):
        current_time = step * dt

        # --- Calculate Control Action Deviations ---
        try:
            # Controller calculates raw deviations: u_dev = -K * (x - x_trim)
            # Returns a dict {'elevator': ele_dev, 'aileron': ail_dev, ...} in ORIGINAL_INPUT_ORDER
            action_dev_dict = controller.compute_action_deviation(current_obs_env, X_TRIM_FULL)
        except Exception as e:
            print(f"ERROR during controller action computation at step {step}: {e}")
            break

        # --- Construct Full Action Array for Environment ---
        # Needs to be in ENV_ACTION_ORDER ['elevator', 'aileron', 'throttle', 'rudder']
        act_array_env = np.zeros(len(ENV_ACTION_ORDER), dtype=np.float32)
        action_total_dict_log = {} # For logging the total applied actions

        try:
            # Calculate TOTAL commands (Trim + LQR Deviation) and Clip
            for input_name in ORIGINAL_INPUT_ORDER: # Iterate through LQR controller's output order
                deviation_cmd = action_dev_dict[input_name]
                trim_cmd = U_TRIM_FULL[ORIGINAL_INPUT_ORDER.index(input_name)]

                total_cmd = trim_cmd + deviation_cmd

                # Clip the TOTAL command based on actuator limits
                low_limit, high_limit = ACTION_LIMITS[input_name]
                clipped_total_cmd = np.clip(total_cmd, low_limit, high_limit)

                # Store the clipped total command for logging
                action_total_dict_log[f"{input_name}_total"] = clipped_total_cmd

                # Place the clipped total command into the correct slot for the environment
                env_action_index = env_act_dict[input_name]
                act_array_env[env_action_index] = clipped_total_cmd

                # Debug: Check clipping
                # if total_cmd != clipped_total_cmd:
                #     print(f"  [Clip] Step {step}: Clipped {input_name}: {total_cmd:.4f} -> {clipped_total_cmd:.4f} (Trim: {trim_cmd:.4f}, Dev: {deviation_cmd:.4f})")


        except (KeyError, IndexError) as e:
            print(f"ERROR: Failed to construct full action array at step {step}: {e}")
            break

        # --- Step Environment ---
        try:
            # act_array_env[1] = 0.0
            # act_array_env[3] = 0.0
            print(f"act_array: {act_array_env}")
            obs_next_env, reward, terminated, truncated, info = env.step(act_array_env)
        except Exception as e:
             print(f"ERROR during env.step at step {step}: {e}")
             break

        # --- Log Data ---
        # Calculate full state deviation again for logging purposes
        x_full_current = current_obs_env[full_state_indices_in_env]
        x_deviation_full = x_full_current - X_TRIM_FULL
        # Use the calculated clipped total actions for logging
        update_history(history, current_time, x_deviation_full, ORIGINAL_STATE_ORDER,
                       action_total_dict_log, ORIGINAL_INPUT_ORDER)

        # --- Update State for Next Iteration ---
        current_obs_env = obs_next_env

        # --- Print Status Periodically ---
        if step % SIM_LOG_INTERVAL == 0 or terminated or truncated:
            print(f"\nStep: {step}, Time: {current_time:.2f}s")
            # Print key deviations
            dev_strs = []
            for name in ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi']:
                idx = ORIGINAL_STATE_ORDER.index(name)
                val = x_deviation_full[idx]
                if name in ['p', 'q', 'r', 'phi', 'theta', 'psi']:
                    dev_strs.append(f"{name}={np.degrees(val):.2f}deg")
                else:
                    dev_strs.append(f"{name}={val:.2f}")
            print(f"  Deviation: [{', '.join(dev_strs)}]")

            # Print total applied actions
            act_strs = []
            for name in ['elevator', 'aileron', 'rudder', 'throttle']:
                total_val = action_total_dict_log[f"{name}_total"]
                if name in ['elevator', 'aileron', 'rudder']:
                     act_strs.append(f"{name}={np.degrees(total_val):.2f}deg")
                else:
                     act_strs.append(f"{name}={total_val:.3f}")
            print(f"  Total Act: [{', '.join(act_strs)}]")
            print(f"  Env Status: Reward={reward:.3f}, Terminated={terminated}, Truncated={truncated}")


        # --- Check for End of Episode ---
        if terminated or truncated:
            print(f"\nEpisode finished after {step + 1} steps at t={current_time+dt:.2f}s.")
            print(f"Reason: Terminated={terminated}, Truncated={truncated}")
            break
    else:
        print(f"\nEpisode reached step limit ({TIME_STEP_LIMIT}) at t={current_time+dt:.2f}s.")

    print("\n" + "="*60)
    print("--- Simulation Loop Finished ---")
    print("="*60)

    # --- 8. Process Results ---
    # Choose which states to plot deviations for
    states_to_plot = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi']
    plot_full_results(history, states_to_plot, ORIGINAL_INPUT_ORDER)
    if SAVE_RESULTS:
        csv_filename = f"lqr_full_history_airspeed{LINEARIZATION_AIRSPEED}.csv"
        save_history_to_csv(history, OUTPUT_DIR, csv_filename)

    # --- Cleanup ---
    print("\nClosing environment.")
    env.close()
    print("\nScript finished.")


# ==============================================================================
# --- Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    # Attempt to register custom environments if the function exists
    try:
        if hasattr(flyer_env, 'register_flyer_envs') and callable(flyer_env.register_flyer_envs):
            flyer_env.register_flyer_envs()
            print("Custom flyer environments registered via flyer_env.register_flyer_envs().")
        else:
            print("Info: flyer_env module found, but register_flyer_envs() function is missing or not callable. Assuming envs registered elsewhere.")
    except NameError:
        print("Warning: 'flyer_env' module not found or failed to import. Cannot register custom envs automatically. Ensure they are registered if needed.")
    except Exception as e:
        print(f"Warning: An error occurred during flyer_env registration: {e}")

    # Run the main simulation function
    try:
        run_simulation()
    except SystemExit as e:
        print(f"\nExecution aborted: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
