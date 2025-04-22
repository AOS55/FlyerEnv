import os
import csv
import math
import matplotlib.pyplot as plt
import flyer_env # Assuming this registers the env
import gymnasium as gym
import numpy as np
import scipy.linalg # Still needed for discretization if using linear model
from typing import Dict, List, Tuple, Optional, Any

# --- MPC Specific Imports ---
# Ensure do-mpc is installed: pip install do-mpc
import do_mpc

# ==============================================================================
# --- Configuration & Constants ---
# ==============================================================================

# --- Simulation Settings ---
TIME_STEP_LIMIT = 1000      # Simulation duration (steps) - Reduced for faster testing
SIM_LOG_INTERVAL = 50       # Print status every N steps
OUTPUT_DIR = "run_output_mpc_full_aircraft_traj" # Directory for results
SAVE_RESULTS = True         # Save history to CSV?

# --- Model & Trim ---
# Choose the TARGET_TRIM_AIRSPEED which corresponds to the A/B matrices
# and the reference trajectory's intended speed.
TARGET_TRIM_AIRSPEED = 80.0  # m/s - Example value (Must match linearization point)
LINEARIZATION_AIRSPEED = TARGET_TRIM_AIRSPEED
print(f"INFO: Linearization Airspeed Assumption: {LINEARIZATION_AIRSPEED} m/s")

# Order of states must match A_full, B_full, and X_TRIM_FULL
ORIGINAL_STATE_ORDER = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']
N_STATES_FULL = len(ORIGINAL_STATE_ORDER) # Should be 12

# Define the FULL trim state vector corresponding to the linearization point
# This is the reference point around which the MPC might operate, OR
# the MPC might directly track the absolute trajectory reference.
# Placeholder values - YOU MUST PROVIDE THE CORRECT VALUES FOR YOUR MODEL/SPEED
X_TRIM_FULL = np.array([
    TARGET_TRIM_AIRSPEED, 0.0, 1.564, 0.0, -4.826e-2, 0.0, # u, v, w, p, q, r
    0.0, 0.020, 0.0,                                     # phi, theta, psi
    0.0, 0.0, -500.0                                        # x, y, z (Absolute position trim often 0 for stabilization)
])
print(f"WARNING: Using Placeholder X_TRIM_FULL = {X_TRIM_FULL}.")
print(f"Ensure ALL components are correct for straight, level flight at {TARGET_TRIM_AIRSPEED} m/s!")
assert len(X_TRIM_FULL) == N_STATES_FULL, f"X_TRIM_FULL length mismatch. Expected {N_STATES_FULL}"


# Order of inputs for U_TRIM_FULL must match B_full
ORIGINAL_INPUT_ORDER = ['elevator', 'aileron', 'rudder', 'throttle']
N_INPUTS_FULL = len(ORIGINAL_INPUT_ORDER) # Should be 4

# Define the FULL trim input vector corresponding to X_TRIM_FULL
# YOU MUST PROVIDE THE CORRECT VALUES FOR YOUR MODEL/SPEED
U_TRIM_FULL = np.array([
    -0.028833359, 0.0, 0.0, 0.77409241 # ele, ail, rud, thr
])
print(f"WARNING: Using Placeholder U_TRIM_FULL = {U_TRIM_FULL}.")
print(f"Ensure components are correct for trim at {TARGET_TRIM_AIRSPEED} m/s!")
assert len(U_TRIM_FULL) == N_INPUTS_FULL, f"U_TRIM_FULL length mismatch. Expected {N_INPUTS_FULL}"

# --- MPC Tuning ---
PREDICTION_HORIZON = 20     # N: Number of steps to predict ahead
# Control horizon (how many steps the input is optimized for, usually 1 for MPC)
# If N_CONTROL_HORIZON < PREDICTION_HORIZON, later inputs are held constant.
N_CONTROL_HORIZON = 1

# Penalties for state deviations FROM REFERENCE TRAJECTORY
# [u, v, w, p, q, r, phi, theta, psi, x, y, z]
# *** THESE ARE EXAMPLES - TUNING IS REQUIRED ***
Q_MPC_DIAG = [
    0.5,    # u (track speed)
    0.1,    # v (minimize sideslip)
    0.1,    # w (minimize vertical speed deviation body frame)
    0.01,   # p (minimize roll rate)
    0.01,   # q (minimize pitch rate)
    0.01,   # r (minimize yaw rate)
    0.1,    # phi (track roll)
    0.1,    # theta (track pitch)
    0.1,    # psi (track heading)
    10.0,   # pos_x (track x position - HIGH PENALTY)
    10.0,   # pos_y (track y position - HIGH PENALTY)
    10.0    # pos_z (track altitude - HIGH PENALTY)
]
Q_MPC = np.diag(Q_MPC_DIAG)

# Penalties for control effort (Applied by set_rterm)
# [elevator, aileron, rudder, throttle]
# *** THESE ARE EXAMPLES - TUNING IS REQUIRED ***
R_MPC_DIAG = [
    1.0,   # elevator
    5.0,   # aileron
    5.0,   # rudder
    0.5    # throttle
]
R_MPC = np.diag(R_MPC_DIAG)

# Penalty for change in control input (delta U) is handled implicitly by do-mpc

# Order expected/provided by the Gymnasium environment `flyer_env`
# !!! VERIFY THIS MATCHES YOUR `flyer_env` !!!
ENV_OBS_ORDER = ['x', 'y', 'z', 'phi', 'theta', 'psi', 'u', 'v', 'w', 'p', 'q', 'r']
ENV_ACTION_ORDER = ['elevator', 'aileron', 'throttle', 'rudder'] # Note: Order might differ

# --- Environment Settings ---
# Make sure this matches your registered env name & task type
ENV_ID = "flyer_trajectory-v1" #  OR "flyer_trajectory-v1" if you have it
# --- IMPORTANT: Configure ENV_ID and gym.make() to use the TRAJECTORY TASK ---
# This example uses ControlFlyerEnv, you need to adapt it for TrajectoryFlyerEnv
# which should provide the reference trajectory.
# Example Trajectory Task Params (MUST MATCH your TrajectoryFlyerEnv constructor)
MOTION_TYPE = "straight_and_level"
MOTION_PARAMS = {"target_distance": 1000.0}
MOTION_TYPE = "coordinated_turn"
MOTION_PARAMS = {"turn_radius": 500.0, "turn_angle": 90.0, "direction": "Right"}
TARGET_VELOCITY = TARGET_TRIM_AIRSPEED # Trajectory speed
start_pos_ned = (X_TRIM_FULL[9], X_TRIM_FULL[10], X_TRIM_FULL[11]) # (x, y, z)
start_heading_deg = np.degrees(X_TRIM_FULL[8]) # psi


ENV_SEED = 42
ENV_RENDER_MODE = "rgb_array" # Use None for faster runs
USE_FULL_AIRCRAFT = True # Must be True for FullAircraftPreset/trim features


# ==============================================================================
# --- Original Full System Matrices (Reference) ---
# ==============================================================================
# State order: u, v, w, p, q, r, phi, theta, psi, pos_x, pos_y, pos_z (12 states)
# Input order: elevator, aileron, rudder, throttle (4 inputs)
# !!! WARNING: These hardcoded matrices MUST correspond to TARGET_TRIM_AIRSPEED !!!
A_cont = np.array([
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
B_cont = np.array([
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
assert A_cont.shape == (N_STATES_FULL, N_STATES_FULL), "A_cont shape mismatch"
assert B_cont.shape == (N_STATES_FULL, N_INPUTS_FULL), "B_cont shape mismatch"

# --- Define Actuator Limits ---
ACTION_LIMITS = {
    'elevator': (math.radians(-25), math.radians(25)),
    'aileron': (math.radians(-20), math.radians(20)),
    'rudder': (math.radians(-30), math.radians(30)),
    'throttle': (0.0, 1.0)
}
# Convert limits to arrays for do-mpc
INPUT_LOWER_BOUNDS = np.array([ACTION_LIMITS[name][0] for name in ORIGINAL_INPUT_ORDER])
INPUT_UPPER_BOUNDS = np.array([ACTION_LIMITS[name][1] for name in ORIGINAL_INPUT_ORDER])

# --- Define State Limits (Optional) ---
# Example: Limit pitch angle
STATE_LOWER_BOUNDS = -np.inf * np.ones(N_STATES_FULL)
STATE_UPPER_BOUNDS = np.inf * np.ones(N_STATES_FULL)
# STATE_LOWER_BOUNDS[ORIGINAL_STATE_ORDER.index('theta')] = math.radians(-30)
# STATE_UPPER_BOUNDS[ORIGINAL_STATE_ORDER.index('theta')] = math.radians(30)


# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def get_env_indices(env_list: List[str], target_list: List[str]) -> List[int]:
    """ Finds indices of target items within the environment list. """
    try:
        indices = [env_list.index(name) for name in target_list]
        return indices
    except ValueError as e:
        missing_item = next((name for name in target_list if name not in env_list), None)
        print(f"ERROR: Item '{missing_item}' in target list {target_list} not found in environment list '{env_list}'.")
        raise

def discretize_state_space(A_cont, B_cont, dt):
    """Discretizes continuous state space matrices A, B using zero-order hold."""
    n_states = A_cont.shape[0]
    n_inputs = B_cont.shape[1]

    # Create augmented matrix M = [[A, B], [0, 0]]
    M = np.zeros((n_states + n_inputs, n_states + n_inputs))
    M[:n_states, :n_states] = A_cont
    M[:n_states, n_states:] = B_cont

    # Compute matrix exponential e^(M*dt)
    try:
        M_exp = scipy.linalg.expm(M * dt)
    except Exception as e:
        print(f"Error computing matrix exponential: {e}")
        # Fallback or error handling
        raise SystemExit("Matrix exponential failed, cannot discretize.")

    # Extract discrete matrices Ad, Bd
    Ad = M_exp[:n_states, :n_states]
    Bd = M_exp[:n_states, n_states:]

    return Ad, Bd

# Placeholder for non-linear dynamics if available from flyer_env
# def nonlinear_dynamics(x, u):
#     """
#     Placeholder for the non-linear aircraft dynamics function.
#     This function should take the current state x and input u,
#     and return the state derivative x_dot.
#     It needs to be implemented based on your specific flyer_env backend.
#     """
#     # Example structure (replace with actual dynamics):
#     # x_dot = flyer_env.calculate_dynamics(x, u) # Hypothetical function
#     # return x_dot
#     raise NotImplementedError("Non-linear dynamics function not implemented.")
#     # If using linear model for MPC, this function isn't strictly needed
#     # but do-mpc is more powerful with the non-linear model.
#     return np.zeros_like(x)


# ==============================================================================
# --- MPC Controller Class ---
# ==============================================================================

class MPCController:
    """
    Model Predictive Controller using do-mpc.
    Handles setup and computes control actions based on predictions.
    """
    def __init__(self,
                 model: do_mpc.model.Model,
                 dt: float, # <<< Added dt parameter
                 prediction_horizon: int,
                 control_horizon: int,
                 cost_q: np.ndarray, # State weighting matrix (for model expression)
                 cost_r: np.ndarray, # Input weighting matrix (for rterm)
                 # cost_p: np.ndarray, # Input rate weighting matrix (REMOVED, handled implicitly)
                 state_indices_in_env: List[int],
                 input_order: List[str],
                 env_input_order: List[str],
                 input_lower_bounds: np.ndarray,
                 input_upper_bounds: np.ndarray,
                 state_lower_bounds: np.ndarray,
                 state_upper_bounds: np.ndarray,
                 x_trim: np.ndarray, # For calculating deviations if needed
                 u_trim: np.ndarray): # For calculating deviations if needed

        self.model = model
        self.dt = dt # <<< Store dt
        self.n_states = model.n_x
        self.n_inputs = model.n_u
        self.n_params = model.n_p # Number of parameters (for reference traj)
        self.pred_horizon = prediction_horizon
        self.ctrl_horizon = control_horizon # Usually 1 for standard MPC

        self.state_indices_in_env = state_indices_in_env
        self.input_order = input_order # Order MPC expects/outputs
        self.env_input_order = env_input_order # Order Environment expects
        self.x_trim = x_trim
        self.u_trim = u_trim

        # --- Assertions ---
        assert cost_q.shape == (self.n_states, self.n_states), "Q shape mismatch"
        assert cost_r.shape == (self.n_inputs, self.n_inputs), "R shape mismatch"
        # assert cost_p.shape == (self.n_inputs, self.n_inputs), "P (deltaU) shape mismatch" # Removed
        assert len(state_indices_in_env) == self.n_states, "State indices mismatch"
        assert len(input_order) == self.n_inputs, "Input order mismatch"
        assert len(input_lower_bounds) == self.n_inputs, "Input lower bounds mismatch"
        assert len(input_upper_bounds) == self.n_inputs, "Input upper bounds mismatch"
        assert len(state_lower_bounds) == self.n_states, "State lower bounds mismatch"
        assert len(state_upper_bounds) == self.n_states, "State upper bounds mismatch"

        print("\n--- Initializing MPC Controller ---")
        print(f"Prediction Horizon (N): {self.pred_horizon}")
        print(f"Control Horizon: {self.ctrl_horizon}")
        print(f"State Indices in Env: {self.state_indices_in_env}")
        print(f"MPC Input Order: {self.input_order}")
        print(f"Env Input Order: {self.env_input_order}")

        # --- Setup MPC Controller ---
        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': self.pred_horizon,
            'n_robust': 0, # No robust MPC in this example
            'open_loop': False,
            't_step': self.dt, # <<< Use stored dt here
            'state_discretization': 'discrete', # Since we provide Ad, Bd
            'store_full_solution': False, # True for debugging
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
        self.mpc.set_param(**setup_mpc)

        # --- Cost Function ---
        # Define the stage cost (lterm) using the expression from the model
        lterm = model.aux['stage_cost'] # Access the expression defined earlier
        # Define the terminal cost (mterm) - often same as lterm or zero
        mterm = model.aux['stage_cost'] # Penalize deviation at the end of horizon too

        # Set the objective for the MPC controller
        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        # --- Penalty on control effort and rate of change ---
        # The keyword MUST match the input variable name ('u' in this case).
        # Pass the R matrix (must be convertible to CasADi DM).
        self.mpc.set_rterm(u=np.array(R_MPC_DIAG))

        # --- Constraints ---
        # Input constraints
        self.mpc.bounds['lower', '_u', 'u'] = input_lower_bounds
        self.mpc.bounds['upper', '_u', 'u'] = input_upper_bounds
        # State constraints (optional)
        # self.mpc.bounds['lower', '_x', ORIGINAL_STATE_ORDER] = state_lower_bounds
        # self.mpc.bounds['upper', '_x', ORIGINAL_STATE_ORDER] = state_upper_bounds

        # --- Set Parameters ---
        # Time-varying parameters (TVP) for reference trajectory
        param_names = [f'x_ref_{i}' for i in range(self.n_states)]
        param_names += [f'u_ref_{i}' for i in range(self.n_inputs)]
        self.p_template = self.mpc.get_p_template(self.pred_horizon) # Not used if using TVP only
        self.tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(t_now):
             # This function provides the structure for TVP.
             # It will be populated with data in the main loop.
             return self.tvp_template

        self.mpc.set_tvp_fun(tvp_fun)

        # --- Final Setup ---
        self.mpc.setup()
        self.mpc.set_initial_guess() # Initialize solver
        print("MPC Controller Initialized.")


    def compute_action(self, current_obs_env: np.ndarray, x_ref_horizon: np.ndarray, u_ref_horizon: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculates the optimal control action using MPC.

        Args:
            current_obs_env: The full observation vector from the environment.
            x_ref_horizon: The reference state trajectory over the prediction horizon (shape: [pred_horizon+1, n_states]).
            u_ref_horizon: The reference input trajectory over the prediction horizon (shape: [pred_horizon, n_inputs]).

        Returns:
            action_env_order: The optimal action clipped and ordered for the environment.
            mpc_info: Dictionary containing MPC solve status and other info.
        """
        # --- Input Validation ---
        if len(current_obs_env) <= max(self.state_indices_in_env):
             raise ValueError(f"Observation vector size ({len(current_obs_env)}) is too small.")
        if x_ref_horizon.shape != (self.pred_horizon + 1, self.n_states):
            raise ValueError(f"x_ref_horizon shape mismatch. Expected ({self.pred_horizon + 1}, {self.n_states}), got {x_ref_horizon.shape}")
        if u_ref_horizon.shape != (self.pred_horizon, self.n_inputs):
             raise ValueError(f"u_ref_horizon shape mismatch. Expected ({self.pred_horizon}, {self.n_inputs}), got {u_ref_horizon.shape}")

        # --- State Extraction ---
        # Get current state in the order MPC expects (ORIGINAL_STATE_ORDER)
        x0 = current_obs_env[self.state_indices_in_env]

        # --- Set Time-Varying Parameters (Reference Trajectory) ---
        # Populate the tvp_template with the reference trajectory for the horizon
        try:
            # Populate x_ref for horizon steps 0 to N
            for k in range(self.pred_horizon + 1):
                self.tvp_template['_tvp', k, 'x_ref'] = x_ref_horizon[k].reshape(-1, 1)

            # Populate u_ref for horizon steps 0 to N-1
            for k in range(self.pred_horizon):
                self.tvp_template['_tvp', k, 'u_ref'] = u_ref_horizon[k].reshape(-1, 1)

            # Set the populated template for the current optimization step
            # This lambda ensures the latest tvp_template is used
            self.mpc.set_tvp_fun(lambda t_now: self.tvp_template)

        except Exception as e:
             print(f"Error setting TVP parameters: {e}")
             u_optimal = self.u_trim # Fallback
             mpc_info = {'status': 'TVP Error'}
             return self._prepare_action_for_env(u_optimal), mpc_info


        # --- Solve MPC Optimization ---
        try:
            # The make_step function solves the OCP and returns the first optimal input
            u_optimal = self.mpc.make_step(x0) # u_optimal is shape (n_inputs,)
            solve_status = 'Success'
        except do_mpc.tools.error.SolverError as e:
            print(f"MPC Solver Error: {e}")
            u_optimal = self.u_trim # Fallback
            solve_status = 'SolverError'
        except Exception as e: # Catch other potential errors during solve
             print(f"Unexpected error during MPC solve: {e}")
             u_optimal = self.u_trim # Fallback
             solve_status = 'OtherSolveError'

        mpc_info = {'status': solve_status}

        # --- Prepare Action for Environment ---
        action_env_order = self._prepare_action_for_env(u_optimal.flatten())

        return action_env_order, mpc_info


    def _prepare_action_for_env(self, u_optimal_mpc_order: np.ndarray) -> np.ndarray:
        """ Clips the optimal action and reorders it for the environment. """
        # Create action dict in MPC order
        action_mpc_dict = {name: u_optimal_mpc_order[i] for i, name in enumerate(self.input_order)}

        # Create empty action array in Environment order
        act_array_env = np.zeros(len(self.env_input_order), dtype=np.float32)

        # Fill environment action array, applying clipping
        for input_name_mpc in self.input_order:
            total_cmd = action_mpc_dict[input_name_mpc]
            low_limit, high_limit = ACTION_LIMITS[input_name_mpc]
            clipped_total_cmd = np.clip(total_cmd, low_limit, high_limit)

            try:
                env_action_index = self.env_input_order.index(input_name_mpc)
                act_array_env[env_action_index] = clipped_total_cmd
            except ValueError:
                print(f"Warning: MPC input '{input_name_mpc}' not found in Env Action Order {self.env_input_order}. Skipping.")

        return act_array_env


# ==============================================================================
# --- Plotting and History Functions (Mostly Unchanged from LQR) ---
# ==============================================================================
# (Plotting, update_history, save_history_to_csv functions remain the same)
def plot_results(history: Dict[str, list], state_plot_order: List[str], input_order: List[str], x_ref_history: Optional[np.ndarray] = None):
    """ Plot key states (absolute or deviation) and inputs. """
    if not history or not history.get('time'):
        print("History is empty, cannot plot.")
        return

    time = history['time']
    if not time:
        print("History has no time entries, cannot plot.")
        return

    print("\n--- Plotting Results ---")
    state_labels = {
        'u': 'u (m/s)', 'v': 'v (m/s)', 'w': 'w (m/s)',
        'p': 'p (rad/s)', 'q': 'q (rad/s)', 'r': 'r (rad/s)',
        'phi': 'Roll phi (rad)', 'theta': 'Pitch theta (rad)', 'psi': 'Yaw psi (rad)',
        'x': 'X Pos (m)', 'y': 'Y Pos (m)', 'z': 'Z Pos (m)'
    }
    input_labels = {
        'elevator': 'Elevator (rad)', 'aileron': 'Aileron (rad)',
        'rudder': 'Rudder (rad)', 'throttle': 'Throttle (%)'
    }

    # --- Plot States (Absolute Values) ---
    n_states_plot = len(state_plot_order)
    n_rows_s = math.ceil(n_states_plot / 3)
    plt.figure(figsize=(18, 5 * n_rows_s))
    plt.suptitle('MPC Control: State Evolution vs Reference', fontsize=16)
    for i, key in enumerate(state_plot_order):
        plt.subplot(n_rows_s, 3, i + 1)
        actual_key = f"{key}_actual"
        if actual_key not in history or not history[actual_key]:
            print(f"Warning: State '{actual_key}' not found or empty.")
            plt.title(f'State: {state_labels.get(key, key)} (No Data)')
            continue

        actual_state = np.array(history[actual_key])
        label_rad = state_labels.get(key, key)
        plot_in_deg = key in ['p', 'q', 'r', 'phi', 'theta', 'psi']

        if plot_in_deg:
            label_deg = label_rad.replace("rad", "deg")
            plt.plot(time, np.degrees(actual_state), label='Actual State', linestyle='-')
            if x_ref_history is not None and x_ref_history.shape[0] == len(time): # Check length match
                ref_idx = ORIGINAL_STATE_ORDER.index(key)
                plt.plot(time, np.degrees(x_ref_history[:, ref_idx]), label='Reference', linestyle=':')
            plt.ylabel(label_deg)
            plt.title(f'State: {label_deg}')
        else:
            plt.plot(time, actual_state, label='Actual State', linestyle='-')
            if x_ref_history is not None and x_ref_history.shape[0] == len(time): # Check length match
                 ref_idx = ORIGINAL_STATE_ORDER.index(key)
                 plt.plot(time, x_ref_history[:, ref_idx], label='Reference', linestyle=':')
            plt.ylabel(label_rad)
            plt.title(f'State: {label_rad}')

        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- Plot Control Inputs (Total Applied) ---
    n_inputs = len(input_order)
    plt.figure(figsize=(6 * n_inputs, 5))
    plt.suptitle('MPC Control: Control Inputs (Applied)', fontsize=16)
    for i, key in enumerate(input_order):
        plt.subplot(1, n_inputs, i + 1)
        total_key = f"{key}_total"
        if total_key not in history or not history[total_key]:
            print(f"Warning: Total input '{total_key}' not found.")
            plt.title(f'Input: {input_labels.get(key, key)} (No Data)')
            continue

        actual_input = np.array(history[total_key])
        label_rad = input_labels.get(key, key)
        plot_in_deg = key in ['elevator', 'aileron', 'rudder']

        if plot_in_deg:
            label_deg = label_rad.replace("rad", "deg")
            plt.plot(time, np.degrees(actual_input))
            plt.ylabel(label_deg)
            plt.title(f'Input: {label_deg}')
        else:
            plt.plot(time, actual_input)
            plt.ylabel(label_rad)
            plt.title(f'Input: {label_rad}')

        # Plot Limits
        low_limit, high_limit = ACTION_LIMITS[key]
        if plot_in_deg:
            plt.axhline(np.degrees(low_limit), color='r', linestyle='--', label='Limit')
            plt.axhline(np.degrees(high_limit), color='r', linestyle='--')
        else:
            plt.axhline(low_limit, color='r', linestyle='--', label='Limit')
            plt.axhline(high_limit, color='r', linestyle='--')

        # Optional: Plot trim input
        trim_val = U_TRIM_FULL[ORIGINAL_INPUT_ORDER.index(key)]
        if plot_in_deg:
            plt.axhline(np.degrees(trim_val), color='g', linestyle=':', label=f'Trim ({np.degrees(trim_val):.2f} deg)')
        else:
            plt.axhline(trim_val, color='g', linestyle=':', label=f'Trim ({trim_val:.3f})')

        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def update_history(history: Dict[str, list], current_time: float,
                   current_state_mpc_order: np.ndarray, state_order: List[str],
                   action_total_dict: Dict[str, float], input_order: List[str]):
    """Appends current step's data to the history dictionary."""
    history['time'].append(current_time)

    # Log actual states (_actual suffix)
    for i, key in enumerate(state_order):
        history.setdefault(f"{key}_actual", []).append(current_state_mpc_order[i])

    # Log total actions applied (_total suffix)
    for key in input_order:
        history.setdefault(f"{key}_total", []).append(action_total_dict.get(key, np.nan))

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
        num_entries = len(history['time'])
        headers = list(history.keys())
        data_columns = []
        for header in headers:
            column = history.get(header, [])
            if len(column) < num_entries:
                print(f"Warning: History column '{header}' length {len(column)} != expected {num_entries}. Padding.")
                column.extend([np.nan] * (num_entries - len(column)))
            data_columns.append(column[:num_entries])

        rows = zip(*data_columns)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Results successfully saved.")

    except Exception as e:
        print(f"ERROR saving results to CSV: {e}")

# ==============================================================================
# --- Main Simulation Function ---
# ==============================================================================

def run_simulation():
    """
    Sets up the Full Aircraft MPC controller and runs the simulation loop.
    """
    print("="*60)
    print("--- Starting Full Aircraft MPC Control Simulation ---")
    print("="*60)
    print(f"Using Linearization Airspeed Assumption: {LINEARIZATION_AIRSPEED} m/s")
    print(f"Reference Full Trim State (Order: {ORIGINAL_STATE_ORDER}):\n {np.round(X_TRIM_FULL, 4)}")
    print(f"Reference Full Trim Input (Order: {ORIGINAL_INPUT_ORDER}):\n {np.round(U_TRIM_FULL, 4)}")

    # --- 1. Setup Environment ---
    print("\n--- Setting up Environment ---")
    print(f"Env ID: {ENV_ID}, Seed: {ENV_SEED}, Render: {ENV_RENDER_MODE}")

    # --- IMPORTANT ---
    # You MUST configure the environment to use the TRAJECTORY task
    # AND provide the necessary motion_type and motion_params.
    # The example below uses ControlFlyerEnv initialization, modify it:
    try:
        # MODIFY THIS to initialize your TrajectoryFlyerEnv
        env = gym.make(ENV_ID,
            seed=ENV_SEED,
            render_mode=ENV_RENDER_MODE,
            # --- Trajectory Params ---
            motion_type=MOTION_TYPE,
            target_velocity=TARGET_VELOCITY,
            motion_params=MOTION_PARAMS,
            start_position=start_pos_ned, # Use fixed start position
            start_heading_deg=start_heading_deg, # Use fixed start heading
            # --- Common Params ---
            use_full_aircraft=USE_FULL_AIRCRAFT, # TrajectoryFlyerEnv might default to Dubins
            max_episode_steps=TIME_STEP_LIMIT
        )

        # Get simulation timestep from env
        try:
            dt = env.unwrapped.dt # Assumes unwrapped env has dt attribute
            print(f"Environment dt: {dt:.6f} s")
        except AttributeError:
            try: # Check config if possible
                dt = env.unwrapped.config["time_step"]
                print(f"Environment dt from config: {dt:.6f} s")
            except (AttributeError, KeyError):
                 dt_fallback = 1/60.0
                 dt = dt_fallback
                 print(f"Warning: Cannot access env dt, assuming dt={dt:.6f} s")

    except Exception as e:
        print(f"ERROR creating environment '{ENV_ID}': {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit("Environment creation failed.")

    # --- 2. Discretize System Matrices (if using linear model for MPC) ---
    Ad, Bd = discretize_state_space(A_cont, B_cont, dt)
    print("\n--- Discretized System Matrices (Ad, Bd) ---")
    # print(f"Ad ({Ad.shape}):\n{np.round(Ad, 3)}") # Optional: Print matrices
    # print(f"Bd ({Bd.shape}):\n{np.round(Bd, 3)}")

    # --- 3. Setup do-mpc Model ---
    print("\n--- Setting up do-mpc Model ---")
    model_type = 'discrete' # Since we provide Ad, Bd
    model = do_mpc.model.Model(model_type)

    # States, Inputs
    x = model.set_variable(var_type='_x', var_name='x', shape=(N_STATES_FULL, 1))
    u = model.set_variable(var_type='_u', var_name='u', shape=(N_INPUTS_FULL, 1)) # Name is 'u'

    # Time-varying parameters (for reference trajectory)
    # Define parameters for x_ref and u_ref
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(N_STATES_FULL, 1))
    u_ref = model.set_variable(var_type='_tvp', var_name='u_ref', shape=(N_INPUTS_FULL, 1)) # Optional: If tracking input ref

    # System Dynamics (Linearized Model)
    # x_{k+1} = Ad * x_k + Bd * u_k
    x_next = Ad @ x + Bd @ u
    model.set_rhs('x', x_next)

    # Define the quadratic cost term for state tracking
    # This expression represents the stage cost (lterm)
    state_tracking_cost = (x - x_ref).T @ Q_MPC @ (x - x_ref)
    # If also tracking input reference u_ref:
    # state_tracking_cost += (u - u_ref).T @ R_MPC @ (u - u_ref) # Add input tracking here if needed

    model.set_expression(expr_name='stage_cost', expr=state_tracking_cost)

    model.setup()


    # --- 4. Get Mappings for Environment ---
    print("\n--- Establishing Environment Mappings ---")
    print(f"MPC State Order (Original): {ORIGINAL_STATE_ORDER}")
    print(f"MPC Input Order (Original): {ORIGINAL_INPUT_ORDER}")
    print(f"Environment Observation Order: {ENV_OBS_ORDER}")
    print(f"Environment Action Order:      {ENV_ACTION_ORDER}")

    try:
        # Indices of MPC states in the *environment observation* vector
        full_state_indices_in_env = get_env_indices(ENV_OBS_ORDER, ORIGINAL_STATE_ORDER)
        print(f"Indices of MPC States in Env Obs: {full_state_indices_in_env}")
        # Create mapping from env action name to index
        env_act_dict = {name: i for i, name in enumerate(ENV_ACTION_ORDER)}
    except ValueError as e:
        print(f"ERROR: Failed to map state/action names: {e}")
        env.close()
        raise SystemExit("Index mapping failed.")

    # --- 5. Instantiate MPC Controller ---
    controller = MPCController(model=model,
                               dt=dt, # Pass dt here
                               prediction_horizon=PREDICTION_HORIZON,
                               control_horizon=N_CONTROL_HORIZON,
                               cost_q=Q_MPC, # Pass Q matrix for reference in expression
                               cost_r=R_MPC, # Pass R matrix for rterm
                               # cost_p=P_MPC, # Pass P matrix for rterm (REMOVED)
                               state_indices_in_env=full_state_indices_in_env,
                               input_order=ORIGINAL_INPUT_ORDER,
                               env_input_order=ENV_ACTION_ORDER,
                               input_lower_bounds=INPUT_LOWER_BOUNDS,
                               input_upper_bounds=INPUT_UPPER_BOUNDS,
                               state_lower_bounds=STATE_LOWER_BOUNDS,
                               state_upper_bounds=STATE_UPPER_BOUNDS,
                               x_trim=X_TRIM_FULL,
                               u_trim=U_TRIM_FULL)


    # --- 6. Initialize Simulation (Reset) ---
    print("\n--- Initializing Simulation ---")
    try:
        obs_env_order_initial, info = env.reset(seed=ENV_SEED)
        print(f"Initial Observation (Env Order, len={len(obs_env_order_initial)}):")
        print(f"  {np.round(obs_env_order_initial, 3)}")
    except Exception as e:
        print(f"ERROR during env.reset(): {e}")
        env.close()
        raise SystemExit("Environment reset failed.")

    # --- Verify observation length ---
    if len(obs_env_order_initial) != len(ENV_OBS_ORDER):
        print(f"ERROR: Observation length mismatch.")
        env.close()
        raise SystemExit("Observation length mismatch.")

    current_obs_env = obs_env_order_initial.copy()

    # Initialize history dictionary
    history_keys = ['time']
    history_keys += [f"{s}_actual" for s in ORIGINAL_STATE_ORDER] # Log actual state
    history_keys += [f"{i}_total" for i in ORIGINAL_INPUT_ORDER] # Log total applied action
    history = {key: [] for key in history_keys}
    # Store reference trajectory for plotting
    x_ref_history = []

    print("\n" + "="*60)
    print(f"--- Starting Simulation Loop ({TIME_STEP_LIMIT} steps) ---")
    print("="*60)

    # --- 7. Simulation Loop ---
    for step in range(TIME_STEP_LIMIT):
        current_time = step * dt

        # --- Get Reference Trajectory ---
        # THIS IS A CRITICAL PLACEHOLDER - You need to get the reference trajectory
        # for the *prediction horizon* (N steps ahead) from your environment setup.
        try:
            # Hypothetical function - Replace with your actual implementation
            # Should return x_ref[N+1, n_states], u_ref[N, n_inputs]
            x_ref_horizon, u_ref_horizon = env.unwrapped.get_reference_trajectory(
                 current_time=current_time,
                 horizon_steps=PREDICTION_HORIZON,
                 dt=dt
            )
            # Store the reference for the *current* step for plotting
            x_ref_current_step = x_ref_horizon[0, :]
            x_ref_history.append(x_ref_current_step)

        except AttributeError:
             print(f"ERROR: env.unwrapped does not have 'get_reference_trajectory' method.")
             print("Please implement a way to retrieve the reference trajectory in your env.")
             # Fallback: Use trim state as constant reference (won't follow trajector
             x_ref_horizon = np.tile(X_TRIM_FULL, (PREDICTION_HORIZON + 1, 1))
             u_ref_horizon = np.tile(U_TRIM_FULL, (PREDICTION_HORIZON, 1))
             x_ref_history.append(X_TRIM_FULL) # Log trim state

        except Exception as e:
            print(f"ERROR getting reference trajectory: {e}")
            break # Stop simulation if reference cannot be obtained


        # --- Calculate Control Action ---
        try:
            # Controller computes the optimal action for the environment
            action_env_order, mpc_info = controller.compute_action(
                current_obs_env, x_ref_horizon, u_ref_horizon
            )
            # print(f"Step {step} MPC Status: {mpc_info['status']}") # DEBUG
            # print(f"Step {step} Action (Env Order): {np.round(action_env_order, 4)}") # DEBUG

        except Exception as e:
            print(f"ERROR during controller action computation at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

        # --- Step Environment ---
        try:
            obs_next_env, reward, terminated, truncated, info = env.step(action_env_order)
        except Exception as e:
            print(f"ERROR during env.step at step {step}: {e}")
            break

        # --- Log Data ---
        # Log current actual state (in MPC order) and applied action
        current_state_mpc_order = current_obs_env[full_state_indices_in_env]
        # Create dict of applied actions for logging
        action_total_dict_log = {name: action_env_order[env_act_dict[name]] for name in ORIGINAL_INPUT_ORDER}
        update_history(history, current_time, current_state_mpc_order, ORIGINAL_STATE_ORDER,
                       action_total_dict_log, ORIGINAL_INPUT_ORDER)

        # --- Update State for Next Iteration ---
        current_obs_env = obs_next_env

        # --- Print Status Periodically ---
        if step % SIM_LOG_INTERVAL == 0 or terminated or truncated:
            print(f"\nStep: {step}, Time: {current_time:.2f}s")
            # Print key actual states vs reference
            state_strs = []
            if x_ref_history: # Check if list is not empty
                current_x_ref = x_ref_history[-1] # Get the reference for the current step
                for name in ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'x', 'y', 'z']:
                    idx = ORIGINAL_STATE_ORDER.index(name)
                    actual_val = current_state_mpc_order[idx]
                    ref_val = current_x_ref[idx]
                    unit = 'deg' if name in ['p', 'q', 'r', 'phi', 'theta', 'psi'] else ''
                    val_str = f"{name}="
                    if unit == 'deg':
                        val_str += f"{np.degrees(actual_val):.2f}({np.degrees(ref_val):.2f}){unit}"
                    else:
                         val_str += f"{actual_val:.2f}({ref_val:.2f}){unit}"
                    state_strs.append(val_str)
                print(f"  State (Ref): [{', '.join(state_strs)}]")
            else:
                print("  State (Ref): [Reference history not available yet]")


            # Print total applied actions
            act_strs = []
            for name in ['elevator', 'aileron', 'rudder', 'throttle']:
                total_val = action_total_dict_log.get(name, np.nan)
                unit = 'deg' if name in ['elevator', 'aileron', 'rudder'] else ''
                if not np.isnan(total_val):
                    if unit == 'deg':
                        act_strs.append(f"{name}={np.degrees(total_val):.2f}{unit}")
                    else:
                        act_strs.append(f"{name}={total_val:.3f}{unit}")
                else:
                    act_strs.append(f"{name}=NaN")

            print(f"  Applied Act: [{', '.join(act_strs)}]")
            print(f"  Env Status: Reward={reward:.3f}, Terminated={terminated}, Truncated={truncated}")


        # --- Check for End of Episode ---
        if terminated or truncated:
            reason = f"Terminated={terminated}, Truncated={truncated}"
            print(f"\nEpisode finished after {step + 1} steps at t={current_time+dt:.2f}s. Reason: {reason}")
            break
    else: # Loop finished without break
        print(f"\nEpisode reached step limit ({TIME_STEP_LIMIT}) at t={current_time+dt:.2f}s.")


    print("\n" + "="*60)
    print("--- Simulation Loop Finished ---")
    print("="*60)

    # --- 8. Process Results ---
    # Convert history list to numpy array for easier indexing
    x_ref_history_np = np.array(x_ref_history) if x_ref_history else None

    # Choose which states to plot
    states_to_plot = ['x', 'y', 'z', 'u', 'phi', 'theta', 'psi', 'p', 'q', 'r']
    # Check if history and reference have data before plotting
    if history['time'] and x_ref_history_np is not None and len(history['time']) == x_ref_history_np.shape[0]:
        plot_results(history, states_to_plot, ORIGINAL_INPUT_ORDER, x_ref_history_np)
    else:
        print("Skipping plotting due to missing history or reference data mismatch.")
        # Optionally plot without reference:
        # plot_results(history, states_to_plot, ORIGINAL_INPUT_ORDER)

    if SAVE_RESULTS:
        csv_filename = f"mpc_full_trajectory_history_airspeed{LINEARIZATION_AIRSPEED}.csv"
        save_history_to_csv(history, OUTPUT_DIR, csv_filename)

    # --- Cleanup ---
    print("\nClosing environment.")
    env.close()
    print("\nScript finished.")


# ==============================================================================
# --- Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    # Attempt to register custom environments
    try:
        if hasattr(flyer_env, 'register_flyer_envs') and callable(flyer_env.register_flyer_envs):
            flyer_env.register_flyer_envs()
            print("Custom flyer environments registered.")
        else:
             print("Info: flyer_env registration function not found/callable.")
    except NameError:
        print("Warning: 'flyer_env' module not found. Ensure envs are registered.")
    except Exception as e:
        print(f"Warning: Error during flyer_env registration: {e}")

    # Run the main simulation
    try:
        run_simulation()
    except SystemExit as e:
        print(f"\nExecution aborted: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
