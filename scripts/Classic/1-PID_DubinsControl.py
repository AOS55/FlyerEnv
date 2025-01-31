import matplotlib.pyplot as plt
import flyer_env
import gymnasium as gym
import numpy as np

from typing import Dict

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 1/60):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.windup_limit = 20.0
        self.prev_output = 0.0
        self.max_rate = 1.0  # Maximum rate of change of control output
        self.reset()

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, error: float) -> float:
        # Update integral term
        self.integral = np.clip(
            self.integral + error * self.dt,
            -self.windup_limit,
            self.windup_limit
        )

        # Calculate derivative term
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # Calculate raw output
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        # Apply rate limiting
        delta = output - self.prev_output
        delta = np.clip(delta, -self.max_rate, self.max_rate)
        output = self.prev_output + delta
        self.prev_output = output

        return output

class DubinsController:
    def __init__(self, control_type: str, target_value: float):
        self.control_type = control_type
        self.target = target_value

        # Initialize PID controllers for different control types
        if control_type == "altitude":
            self.pid = PIDController(kp=2.0, ki=0.1, kd=0.3)
        elif control_type == "heading":
            self.pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        elif control_type == "speed":
            self.pid = PIDController(kp=3.0, ki=0.1, kd=0.3)

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def __call__(self, obs: Dict[str, float]) -> Dict[str, float]:
        if self.control_type == "altitude":
            # Altitude control using vertical speed
            error = self.target - obs['altitude']

            max_error_rate = 100  # m per second
            error = np.clip(error, -max_error_rate, max_error_rate)

            vertical_speed = self.pid.update(error)

            # Clamp vertical speed to reasonable values
            vertical_speed = np.clip(vertical_speed, -10.0, 10.0)

            return {
                'vertical_speed': vertical_speed,
                'bank_angle': 0.0,  # Maintain level flight
                'acceleration': 0.0  # Maintain current speed
            }

        elif self.control_type == "heading":
            # Get heading error considering angle wrapping
            current_heading = obs['heading']
            error = self.normalize_angle(self.target - current_heading)
            bank_angle = self.pid.update(error)

            # Clamp bank angle to reasonable values
            bank_angle = np.clip(bank_angle, -np.pi/4, np.pi/4)

            return {
                'vertical_speed': 0.0,  # Maintain altitude
                'bank_angle': bank_angle,
                'acceleration': 0.0  # Maintain current speed
            }

        elif self.control_type == "speed":
            error = self.target - obs['airspeed']
            acceleration = self.pid.update(error)

            # Clamp acceleration to reasonable values
            acceleration = np.clip(acceleration, -10.0, 10.0)

            return {
                'vertical_speed': 0.0,  # Maintain altitude
                'bank_angle': 0.0,  # Maintain level flight
                'acceleration': acceleration
            }

        return {'vertical_speed': 0.0, 'bank_angle': 0.0, 'acceleration': 0.0}

def plot_tracking(data, control_type: str, target_value=500.0, tolerance=10.0):
    """
    Plot the aircraft state and control inputs.

    Args:
        data (dict): Current observation/action containing state variables
        target_altitude (float): Target altitude in feet
        tolerance (float): Allowed deviation from target
    """


    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.tight_layout(pad=3.0)

    # Plot 1: State Tracking
    ax1.set_title(f'{control_type.capitalize()} Tracking')
    state_var = control_type
    if control_type == "speed":
        state_var = "airspeed"

    ax1.plot(data[state_var], 'g-', label=f'Actual {control_type.capitalize()}')
    ax1.axhline(y=target_value, color='r', linestyle='--', label='Target')
    ax1.fill_between([0, len(data[state_var])],
                        target_value - tolerance,
                        target_value + tolerance,
                        color='r', alpha=0.1)
    ax1.set_ylabel(f'{control_type.capitalize()} Value')
    ax1.set_xlabel('Time Step')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Control Inputs
    ax2.set_title('Control Inputs')
    ax2.plot(data['vertical_speed'], 'b-', label='Vertical Speed')
    ax2.plot(data['bank_angle'], 'g-', label='Bank Angle')
    ax2.plot(data['acceleration'], 'r-', label='Acceleration')
    ax2.set_ylabel('Control Values')
    ax2.set_xlabel('Time Step')
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Reward
    ax3.set_title('Reward')
    ax3.plot(data['reward'], 'purple', label='Reward')
    ax3.set_ylabel('Reward Value')
    ax3.set_xlabel('Time Step')
    ax3.grid(True)
    ax3.legend()

    plt.show()

def update_history(obs_history: Dict[str, list], obs: Dict[str, float], act: Dict[str, float], reward: float) -> Dict[str, list]:
    # Update all state variables
    for key in obs:
        if key not in obs_history:
            obs_history[key] = []
        obs_history[key].append(obs[key])

    # Update control inputs
    for key in act:
        if key not in obs_history:
            obs_history[key] = []
        obs_history[key].append(act[key])

    if 'reward' not in obs_history:
        obs_history['reward'] = []
    obs_history['reward'].append(reward)

    return obs_history

def main():

    # Control task parameters
    scenarios = [
            {"control_type": "altitude", "target_value": 500.0, "tolerance": 10.0},
            {"control_type": "heading", "target_value": np.pi/2, "tolerance": 0.1},  # 90 degrees in radians
            {"control_type": "speed", "target_value": 100.0, "tolerance": 5.0},  # 100 m/s
        ]

    for scenario in scenarios:
        print(f"\Running {scenario['control_type']} control:")
        print(f"Target: {scenario['target_value']}, Tolerance: {scenario['tolerance']}")

        env = gym.make("flyer_control-v1",
            seed=42,
            render_mode="rgb_array",
            start_deviation=100.0,
            control_type=scenario['control_type'],
            target_value=scenario['target_value'],
            tolerance=scenario['tolerance']
        )

        controller = DubinsController(scenario['control_type'], scenario['target_value'])

        obs, info = env.reset()
        obs_history = {}

        for _ in range(1000):
            # Get observation dict
            obs_dict = env.unwrapped._observation.to_dict(obs)

            # Get control action
            act_dict = controller(obs_dict)
            act = env.unwrapped._action.from_dict(act_dict)
            # print(f"Raw action vector: {act}")

            # Step environment
            obs, reward, truncated, terminated, info = env.step(act)

            # Update history
            obs_history = update_history(
                obs_history,
                obs_dict,
                act_dict,
                reward
            )

            if terminated or truncated:
                break

        plot_tracking(obs_history, scenario['control_type'], scenario['target_value'], scenario['tolerance'])

if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    main()
