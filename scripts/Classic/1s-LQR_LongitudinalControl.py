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
SIM_LOG_INTERVAL = 50      # Print status every N steps
OUTPUT_DIR = "run_output_lqr_longitudinal_debug" # Directory for results
SAVE_RESULTS = True        # Save history to CSV?

# --- Model & Trim ---
# <<< CHOOSE THE TARGET TRIM AIRSPEED HERE >>>
# This airspeed MUST match the conditions for which A/B/X_trim were derived.
TARGET_TRIM_AIRSPEED = 80.0  # m/s - Example value

# Use the chosen target airspeed for linearization assumption
LINEARIZATION_AIRSPEED = TARGET_TRIM_AIRSPEED
print(f"INFO: Setting Linearization Airspeed Assumption to: {LINEARIZATION_AIRSPEED} m/s")

# !!! IMPORTANT !!!
# The A_full, B_full matrices AND X_TRIM_LON below MUST correspond
# to the TARGET_TRIM_AIRSPEED chosen above for the Twin Otter model used in flyer_env.
# The current hardcoded values ARE LIKELY INCORRECT for 80.0 m/s.
# You NEED to obtain the correct A/B matrices and the corresponding
# [u, w, q, theta] trim values for TARGET_TRIM_AIRSPEED.

# Define the longitudinal trim state [u, w, q, theta] corresponding to the linearization point
# These are the target values the LQR will try to hold.
X_TRIM_LON = np.array([
    TARGET_TRIM_AIRSPEED,  # u_trim (m/s) - SET TO TARGET AIRSPEED
    1.564,                 # w_trim (m/s) - NEEDS UPDATE FOR 80 m/s
    -4.826e-2,             # q_trim (rad/s) - NEEDS UPDATE FOR 80 m/s
    0.020                  # theta_trim (rad) - NEEDS UPDATE FOR 80 m/s
])
print(f"WARNING: Using X_TRIM_LON = {X_TRIM_LON}. Ensure w, q, theta components are correct for {TARGET_TRIM_AIRSPEED} m/s!")


# Assume trim for lateral controls (will be held constant)
TRIM_ELEVATOR = -0.028833359
TRIM_THROTTLE = 0.77409241
TRIM_AILERON = 0.0  # rad
TRIM_RUDDER = 0.0   # rad

# --- LQR Tuning ---
# Penalties for state deviations [u, w, q, theta]
Q_LON_DIAG = [1.0, 0.0, 0.01, 0.01]
Q_SCALING = 1.0
# Penalties for control effort [elevator, throttle]
R_LON_DIAG = [10.0, 1.0]
R_SCALING = 5.0

# --- State and Input Order Definitions ---
# Order in the original full A/B matrices provided
ORIGINAL_STATE_ORDER = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'pos_x', 'pos_y', 'pos_z']
ORIGINAL_INPUT_ORDER = ['elevator', 'aileron', 'rudder', 'throttle']

# Desired states/inputs for the reduced longitudinal model
LON_STATE_ORDER = ['u', 'w', 'q', 'theta']
LON_INPUT_ORDER = ['elevator', 'throttle']

# Order expected/provided by the Gymnasium environment `flyer_env`
# !!! VERIFY THIS MATCHES YOUR `flyer_env` !!!
ENV_OBS_ORDER = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'u', 'v', 'w', 'p', 'q', 'r']
ENV_ACTION_ORDER = ['elevator', 'aileron', 'throttle', 'rudder']

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
N_STATES_FULL = A_full.shape[0]
N_INPUTS_FULL = B_full.shape[1]

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
def extract_longitudinal_model(
    A_f: np.ndarray, B_f: np.ndarray,
    full_state_order: List[str], full_input_order: List[str],
    lon_state_order: List[str], lon_input_order: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Extracts the longitudinal subsystem matrices (A_lon, B_lon) and indices
    from the full system matrices based on specified state/input names.
    """
    print("\n--- Extracting Longitudinal Model ---")
    print(f"Full System: {A_f.shape[0]} states, {B_f.shape[1]} inputs")
    print(f"Full State Order: {full_state_order}")
    print(f"Full Input Order: {full_input_order}")
    print(f"Desired Lon States: {lon_state_order}")
    print(f"Desired Lon Inputs: {lon_input_order}")

    try:
        # Find indices in the *original* full system order
        lon_state_indices = [full_state_order.index(s) for s in lon_state_order]
        lon_input_indices = [full_input_order.index(i) for i in lon_input_order]
    except ValueError as e:
        print(f"ERROR: State or input name mismatch during extraction: {e}")
        raise

    print(f"Lon State Indices (in Full A/B): {lon_state_indices}")
    print(f"Lon Input Indices (in Full A/B): {lon_input_indices}")

    # Use np.ix_ for advanced indexing to select rows and columns
    A_lon = A_f[np.ix_(lon_state_indices, lon_state_indices)]
    B_lon = B_f[np.ix_(lon_state_indices, lon_input_indices)]

    # --- Verification ---
    n_states_lon = len(lon_state_order)
    n_inputs_lon = len(lon_input_order)
    assert A_lon.shape == (n_states_lon, n_states_lon), \
        f"Extracted A_lon shape mismatch: expected ({n_states_lon},{n_states_lon}), got {A_lon.shape}"
    assert B_lon.shape == (n_states_lon, n_inputs_lon), \
        f"Extracted B_lon shape mismatch: expected ({n_states_lon},{n_inputs_lon}), got {B_lon.shape}"
    print(f"Extracted A_lon ({A_lon.shape}):\n{A_lon}")
    print(f"Extracted B_lon ({B_lon.shape}):\n{B_lon}")

    return A_lon, B_lon, lon_state_indices, lon_input_indices


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
        print(f"\nInverse of R ({R_inv.shape}):\n{R_inv}")
    except np.linalg.LinAlgError:
        print("ERROR: R matrix is singular, cannot compute inverse.")
        raise SystemExit("Cannot calculate LQR gain with singular R matrix.")

    B_transpose = B.T
    print(f"\nTranspose of B ({B_transpose.shape}):\n{B_transpose}")

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
        print(f"ERROR: Item '{e}' in target list not found in environment list '{env_list}'.")
        raise

# ==============================================================================
# --- Longitudinal LQR Controller Class ---
# ==============================================================================
class LongitudinalLQRController:
    """
    Calculates longitudinal control actions (elevator, throttle) using a
    pre-computed reduced-order LQR gain matrix K_lon.
    Requires the full observation vector, longitudinal trim state vector,
    and indices mapping env observation to longitudinal states.
    """
    def __init__(self, K_lon: np.ndarray, lon_state_indices_in_env: List[int], lon_input_order: List[str]):
        self.K_lon = K_lon
        self.n_states_lon = K_lon.shape[1] # Should be 4
        self.n_inputs_lon = K_lon.shape[0] # Should be 2
        self.lon_state_indices_in_env = lon_state_indices_in_env
        self.lon_input_order = lon_input_order # ['elevator', 'throttle']

        assert len(lon_state_indices_in_env) == self.n_states_lon, "Mismatch between K_lon states and provided indices"
        assert len(lon_input_order) == self.n_inputs_lon, "Mismatch between K_lon inputs and input order list"

        # Define action limits (adjust as needed)
        self.action_limits_low = {
            'elevator': math.radians(-25),
            'throttle': 0.0
        }
        self.action_limits_high = {
            'elevator': math.radians(25),
            'throttle': 1.0
        }
        print("\n--- Controller Initialized ---")
        print(f"Using K_lon gain matrix ({self.K_lon.shape})")
        print(f"Extracting states at indices {self.lon_state_indices_in_env} from environment observation.")
        print(f"Expecting trim state vector of size {self.n_states_lon}.")
        print(f"Outputting actions: {self.lon_input_order}")
        print(f"Action limits (low): {self.action_limits_low}")
        print(f"Action limits (high): {self.action_limits_high}")


    def compute_action(self, obs_env_order: np.ndarray, x_trim_lon: np.ndarray) -> Dict[str, float]:
        """
        Calculates the elevator and throttle commands based on current observation
        and longitudinal trim state.
        """
        # --- Input Validation ---
        if len(obs_env_order) < max(self.lon_state_indices_in_env) + 1:
             raise ValueError(f"Observation vector size ({len(obs_env_order)}) is too small. Need at least {max(self.lon_state_indices_in_env) + 1} elements based on indices {self.lon_state_indices_in_env}.")
        if len(x_trim_lon) != self.n_states_lon:
             raise ValueError(f"Longitudinal trim state size ({len(x_trim_lon)}) mismatch. Expected {self.n_states_lon}.")

        # --- State Extraction ---
        # Get the current values of the longitudinal states [u, w, q, theta] from the full observation vector
        x_lon_current = obs_env_order[self.lon_state_indices_in_env]
        # print(f"  [Ctrl] Extracted x_lon_current: {x_lon_current}") # DEBUG

        # --- Calculate State Deviation ---
        # Deviation = Current State - Trim State
        x_deviation_lon = x_lon_current - x_trim_lon
        # print(f"  [Ctrl] Trim state x_trim_lon:      {x_trim_lon}") # DEBUG
        # print(f"  [Ctrl] State Deviation x_dev_lon: {x_deviation_lon}") # DEBUG

        # --- LQR Control Law ---
        # Control Action = -K * Deviation
        # u_lon_raw is shape (2,) -> [elevator_raw, throttle_raw]
        u_lon_raw = -self.K_lon @ x_deviation_lon

        # --- Apply Action Limits (Saturation) ---
        action_dict_lon = {}
        for i, input_name in enumerate(self.lon_input_order):
            raw_action = u_lon_raw[i]
            low_limit = self.action_limits_low[input_name]
            high_limit = self.action_limits_high[input_name]
            clipped_action = np.clip(raw_action, low_limit, high_limit)
            action_dict_lon[input_name] = clipped_action
            # if raw_action != clipped_action: # DEBUG
            #     print(f"  [Ctrl] Clipped {input_name}: {raw_action:.4f} -> {clipped_action:.4f}") # DEBUG

        # print(f"  [Ctrl] Final action_dict_lon: {action_dict_lon}") # DEBUG
        return action_dict_lon


# ==============================================================================
# --- Plotting and History Functions ---
# ==============================================================================
# plot_lon_results(...)    # No changes needed
# update_history(...)      # No changes needed
# save_history_to_csv(...) # No changes needed
# (Keep the existing plotting/history functions as they were)
def plot_lon_results(history: Dict[str, list], lon_state_order: List[str], lon_input_order: List[str]):
    """ Plot longitudinal state deviations and inputs. """
    if not history or not history.get('time'):
        print("History is empty, cannot plot.")
        return

    time = history['time']
    if not time: # Check if time list itself is empty
         print("History has no time entries, cannot plot.")
         return

    print("\n--- Plotting Results ---")

    # Define labels for plots
    state_labels = {'u': 'u (m/s)', 'w': 'w (m/s)', 'q': 'q (rad/s)', 'theta': 'theta (rad)'}
    input_labels = {'elevator': 'elevator (rad)', 'throttle': 'throttle'}

    # --- Plot Longitudinal State Deviations ---
    n_states = len(lon_state_order)
    plt.figure(figsize=(12, 5 * math.ceil(n_states / 2)))
    plt.suptitle('Longitudinal LQR Control: State Deviation Evolution (Target = 0)', fontsize=16)
    for i, key in enumerate(lon_state_order):
        plt.subplot(math.ceil(n_states / 2), 2, i + 1)
        if key not in history or not history[key]:
            print(f"Warning: State deviation '{key}' not found or empty in history.")
            plt.title(f'State Deviation: {state_labels.get(key, key)} (No Data)')
            continue

        actual_deviation = np.array(history[key])
        label_rad = state_labels.get(key, key)

        # Plot in degrees if it's an angle/rate typically viewed in degrees
        if key in ['q', 'theta', 'phi', 'psi', 'p', 'r']:
             label_deg = label_rad.replace("rad", "deg")
             plt.plot(time, np.degrees(actual_deviation), label='Actual Deviation')
             plt.ylabel(label_deg)
             plt.title(f'State Deviation: {label_deg}')
        else:
             plt.plot(time, actual_deviation, label='Actual Deviation')
             plt.ylabel(label_rad)
             plt.title(f'State Deviation: {label_rad}')

        plt.axhline(0.0, color='r', linestyle='--', label='Target (Zero Deviation)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent title overlap

    # --- Plot Longitudinal Inputs ---
    n_inputs = len(lon_input_order)
    plt.figure(figsize=(6 * n_inputs, 5))
    plt.suptitle('Longitudinal LQR Control: Control Inputs', fontsize=16)
    for i, key in enumerate(lon_input_order):
        plt.subplot(1, n_inputs, i + 1)
        if key not in history or not history[key]:
            print(f"Warning: Input '{key}' not found or empty in history.")
            plt.title(f'Input: {input_labels.get(key, key)} (No Data)')
            continue

        actual_input = np.array(history[key])
        label_rad = input_labels.get(key, key)

        if key == 'elevator': # Example: Plot elevator in degrees
             label_deg = label_rad.replace("rad", "deg")
             plt.plot(time, np.degrees(actual_input))
             plt.ylabel(label_deg)
             plt.title(f'Input: {label_deg}')
        else: # Plot others (throttle) as is
             plt.plot(time, actual_input)
             plt.ylabel(label_rad)
             plt.title(f'Input: {label_rad}')

        plt.xlabel('Time (s)')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def update_history(history: Dict[str, list], current_time: float,
                   state_deviation_lon: np.ndarray, lon_state_order: List[str],
                   lon_act_dict: Dict[str, float], lon_input_order: List[str]):
    """Appends current step's longitudinal data to the history dictionary."""
    history['time'].append(current_time)

    # Log longitudinal state deviations
    for i, key in enumerate(lon_state_order):
        history.setdefault(key, []).append(state_deviation_lon[i])

    # Log longitudinal actions applied
    for key in lon_input_order:
        # Use .get to handle potential missing keys gracefully, though should not happen here
        history.setdefault(key, []).append(lon_act_dict.get(key, np.nan))

    # Note: We are only logging longitudinal data here.


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
            # Pad with NaN if a column is unexpectedly shorter (shouldn't happen with update_history)
            if len(column) < num_entries:
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
    Sets up the LQR controller and runs the simulation loop using the simplified
    environment configuration approach.
    """
    print("="*60)
    print("--- Starting Longitudinal LQR Control Simulation ---")
    print("="*60)
    print(f"Using Linearization Airspeed Assumption: {LINEARIZATION_AIRSPEED} m/s")
    print(f"Target Longitudinal Trim State [u, w, q, theta]: {X_TRIM_LON}")
    print(f"Holding Lateral Controls Trim [ail, rud]: [{TRIM_AILERON}, {TRIM_RUDDER}]")
    print(f"Attempting to start simulation at speed: {TARGET_TRIM_AIRSPEED} m/s") # Using TARGET_TRIM_AIRSPEED here

    # --- 1. Extract Longitudinal Model ---
    A_lon, B_lon, _, _ = extract_longitudinal_model(
        A_full, B_full,
        ORIGINAL_STATE_ORDER, ORIGINAL_INPUT_ORDER,
        LON_STATE_ORDER, LON_INPUT_ORDER
    )
    n_states_lon = A_lon.shape[0]
    n_inputs_lon = B_lon.shape[1]

    # --- 2. Design LQR Controller ---
    Q_lon = np.diag(Q_LON_DIAG) * Q_SCALING
    R_lon = np.diag(R_LON_DIAG) * R_SCALING
    K_lon = calculate_lqr_gain(A_lon, B_lon, Q_lon, R_lon)

    # --- 3. Setup Environment (Simplified Approach) ---
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
                       control_type="speed",             # Use speed control logic for start setup
                       target_value=TARGET_TRIM_AIRSPEED, # Set the target start speed
                       start_deviation=0.0,              # Set zero deviation for specific start speed
                       trim_condition=env_trim_condition,  # Configure aircraft model for trim
                       # --- Other necessary arguments ---
                       use_full_aircraft=True,           # Use the full aircraft model (required for trim)
                       episode_length=TIME_STEP_LIMIT,   # Sync episode length
                       tolerance=1.0,                    # Tolerance for speed task (low value, less relevant for LQR)
                       # simplified_spaces=False # Implicitly False when use_full_aircraft=True
                       )

        # Get simulation timestep
        try:
            dt = env.unwrapped.dt
            print(f"Environment dt: {dt:.6f} s")
        except AttributeError:
            # Estimate dt if not available
            # Check if config is accessible after gym.make (might not be standard)
            dt_fallback = 1/60.0
            try:
                 dt_fallback = env.unwrapped.config.get("time_step", dt_fallback)
            except AttributeError:
                 pass # Stick with 1/60 if config not found
            dt = dt_fallback
            print(f"Warning: Cannot access env.unwrapped.dt, assuming dt={dt:.6f} s")

    except Exception as e:
        print(f"ERROR creating environment '{ENV_ID}': {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit("Environment creation failed.")

    # --- 4. Get Mappings for Environment ---
    print("\n--- Establishing Environment Mappings ---")
    print(f"Environment Observation Order: {ENV_OBS_ORDER}")
    print(f"Environment Action Order:      {ENV_ACTION_ORDER}")

    # Indices needed by the controller: Indices of [u, w, q, theta] in the *environment observation* vector
    try:
        env_obs_dict = {name: i for i, name in enumerate(ENV_OBS_ORDER)}
        env_act_dict = {name: i for i, name in enumerate(ENV_ACTION_ORDER)}
        lon_state_indices_in_env = get_env_indices(ENV_OBS_ORDER, LON_STATE_ORDER)
        print(f"Indices of Lon States ({LON_STATE_ORDER}) in Env Obs: {lon_state_indices_in_env}")
    except (ValueError, KeyError) as e:
        print(f"ERROR: Failed to map state/action names to environment indices: {e}")
        print("Ensure ENV_OBS_ORDER and ENV_ACTION_ORDER match the environment's specification.")
        env.close()
        raise SystemExit("Index mapping failed.")

    # --- 5. Instantiate Controller ---
    controller = LongitudinalLQRController(K_lon, lon_state_indices_in_env, LON_INPUT_ORDER)

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
    history_keys = ['time'] + LON_STATE_ORDER + LON_INPUT_ORDER
    history = {key: [] for key in history_keys}
    print(f"Initialized history log for keys: {history_keys}")

    print(f"\nStarting simulation from environment's reset state (attempted start speed {TARGET_TRIM_AIRSPEED} m/s).")
    print(f"LQR controller will act on deviations from trim: {X_TRIM_LON}")


    print("\n" + "="*60)
    print(f"--- Starting Simulation Loop ({TIME_STEP_LIMIT} steps) ---")
    print("="*60)

    # --- 7. Simulation Loop ---
    for step in range(TIME_STEP_LIMIT):
        current_time = step * dt

        # --- Calculate Longitudinal Action ---
        try:
            lon_act_dict = controller.compute_action(current_obs_env, X_TRIM_LON)
        except Exception as e:
            print(f"ERROR during controller action computation at step {step}: {e}")
            break

        # --- Construct Full Action Array for Environment ---
        act_array_env = np.zeros(len(ENV_ACTION_ORDER), dtype=np.float32)
        try:
            # Get LQR deviation commands
            lqr_elevator_cmd = lon_act_dict['elevator']
            lqr_throttle_cmd = lon_act_dict['throttle']

            # --- Calculate TOTAL commands (Trim + LQR) ---
            total_elevator_cmd = TRIM_ELEVATOR + lqr_elevator_cmd
            total_throttle_cmd = TRIM_THROTTLE + lqr_throttle_cmd

            # --- Apply Saturation to TOTAL commands ---
            # Define limits (can reuse from controller or define here)
            ELEVATOR_LIMIT_LOW = math.radians(-25)
            ELEVATOR_LIMIT_HIGH = math.radians(25)
            THROTTLE_LIMIT_LOW = 0.0
            THROTTLE_LIMIT_HIGH = 1.0

            total_elevator_cmd = np.clip(total_elevator_cmd, ELEVATOR_LIMIT_LOW, ELEVATOR_LIMIT_HIGH)
            total_throttle_cmd = np.clip(total_throttle_cmd, THROTTLE_LIMIT_LOW, THROTTLE_LIMIT_HIGH)


            # Fill lateral controls with trim values (usually 0)
            act_array_env[env_act_dict['aileron']] = TRIM_AILERON
            act_array_env[env_act_dict['rudder']] = TRIM_RUDDER

            # Fill longitudinal controls with TOTAL commands
            act_array_env[env_act_dict['elevator']] = total_elevator_cmd
            act_array_env[env_act_dict['throttle']] = total_throttle_cmd

        except (KeyError, IndexError) as e:
            print(f"ERROR: Failed to construct full action array at step {step}: {e}")
            break

        # --- Step Environment ---
        try:
            obs_next_env, reward, terminated, truncated, info = env.step(act_array_env)
        except Exception as e:
             print(f"ERROR during env.step at step {step}: {e}")
             break

        # --- Log Data ---
        x_lon_current = current_obs_env[lon_state_indices_in_env]
        x_deviation_lon = x_lon_current - X_TRIM_LON
        update_history(history, current_time, x_deviation_lon, LON_STATE_ORDER, lon_act_dict, LON_INPUT_ORDER)

        # --- Update State for Next Iteration ---
        current_obs_env = obs_next_env

        # --- Print Status Periodically ---
        if step % SIM_LOG_INTERVAL == 0 or terminated or truncated:
            print(f"\nStep: {step}, Time: {current_time:.2f}s")
            dev_str = ", ".join([f"{name}={val:.2f}" if name not in ['q', 'theta'] else f"{name}={np.degrees(val):.2f}deg" for name, val in zip(LON_STATE_ORDER, x_deviation_lon)])
            print(f"  Lon Deviation: [{dev_str}]")
            act_str = f"ele={np.degrees(lon_act_dict['elevator']):.2f}deg, thr={lon_act_dict['throttle']:.3f}"
            print(f"  Lon Action:    [{act_str}]")
            try:
                phi_dev = current_obs_env[env_obs_dict['phi']] - 0.0
                psi_dev = current_obs_env[env_obs_dict['psi']] - 0.0
                print(f"  Lateral Check: phi_dev={np.degrees(phi_dev):.2f}deg, psi_dev={np.degrees(psi_dev):.2f}deg")
            except (KeyError, IndexError):
                 print("  Lateral Check: Could not retrieve phi/psi from observation.")
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
    plot_lon_results(history, LON_STATE_ORDER, LON_INPUT_ORDER)
    if SAVE_RESULTS:
        csv_filename = f"lqr_lon_history_airspeed{LINEARIZATION_AIRSPEED}.csv"
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
