import matplotlib.pyplot as plt
import flyer_env
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 1/60):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.windup_limit = 20.0
        self.prev_output = 0.0
        self.max_rate = 1.0
        self.reset()

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, error: float) -> float:
        self.integral = np.clip(
            self.integral + error * self.dt,
            -self.windup_limit,
            self.windup_limit
        )
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = (self.kp * error +
                 self.ki * self.integral +
                 self.kd * derivative)

        delta = output - self.prev_output
        delta = np.clip(delta, -self.max_rate, self.max_rate)
        output = self.prev_output + delta
        self.prev_output = output

        return output

class GoalNavigationController:
    """Simple controller for navigating to 3D goal positions."""
    def __init__(self):
        # PID controllers for different aspects of navigation
        self.altitude_pid = PIDController(kp=2.0, ki=0.1, kd=0.3)
        self.heading_pid = PIDController(kp=1.0, ki=0.1, kd=0.5)
        self.speed_pid = PIDController(kp=3.0, ki=0.1, kd=0.3)

        # Desired cruise speed
        self.cruise_speed = 75.0  # m/s

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def get_desired_heading(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Calculate desired heading to goal."""
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        return np.arctan2(dy, dx)

    def __call__(self, obs: Dict[str, float], goal_position: np.ndarray) -> Dict[str, float]:
        # Extract current position
        current_pos = np.array([
            obs['x'],
            obs['y'],
            -obs['altitude']  # Convert to NED frame
        ])

        # Get desired heading to goal
        desired_heading = self.get_desired_heading(current_pos, goal_position)

        # Calculate heading error
        heading_error = self.normalize_angle(desired_heading - obs['heading'])
        bank_angle = self.heading_pid.update(heading_error)
        bank_angle = np.clip(bank_angle, -np.pi/4, np.pi/4)  # Limit bank angle

        # Altitude control
        altitude_error = -goal_position[2] - current_pos[2]  # Convert back from NED
        vertical_speed = self.altitude_pid.update(altitude_error)
        vertical_speed = np.clip(vertical_speed, -10.0, 10.0)

        # Speed control - maintain cruise speed
        speed_error = self.cruise_speed - obs['airspeed']
        acceleration = self.speed_pid.update(speed_error)
        acceleration = np.clip(acceleration, -10.0, 10.0)

        return {
            'vertical_speed': vertical_speed,
            'bank_angle': bank_angle,
            'acceleration': acceleration
        }

def plot_goal_navigation(data: Dict[str, list], goal_position: np.ndarray):
    """Plot the aircraft trajectory and states during goal navigation."""
    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(data['x'], data['y'], data['altitude'], 'b-', label='Aircraft Path')
    ax1.scatter(goal_position[0], goal_position[1], -goal_position[2],
                color='r', marker='*', s=100, label='Goal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.legend()
    ax1.set_title('3D Trajectory')

    # Top-down view
    ax2 = fig.add_subplot(222)
    ax2.plot(data['x'], data['y'], 'b-', label='Path')
    ax2.scatter(goal_position[0], goal_position[1],
                color='r', marker='*', s=100, label='Goal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    ax2.set_title('Top-down View')

    # Control inputs
    ax3 = fig.add_subplot(223)
    ax3.plot(data['vertical_speed'], 'b-', label='Vertical Speed')
    ax3.plot(data['bank_angle'], 'g-', label='Bank Angle')
    ax3.plot(data['acceleration'], 'r-', label='Acceleration')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Values')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Control Inputs')

    # Reward plot
    ax4 = fig.add_subplot(224)
    ax4.plot(data['reward'], 'purple', label='Reward')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title('Reward')

    plt.tight_layout()
    plt.show()

def update_history(obs_history: Dict[str, list], obs: Dict[str, float],
                  act: Dict[str, float], reward: float) -> Dict[str, list]:
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
    # Create environment with specific goal parameters
    env = gym.make('flyer_goal-v1',
        seed=42,
        render_mode="rgb_array",
        distance_range=(1000.0, 1000.0),  # Fixed distance for demo
        altitude_range=(-750.0, -750.0),   # Fixed altitude for demo
        heading_range=(np.pi/2, np.pi/2),  # Fixed heading for demo
        tolerance=50.0,
        reward_type="dense"
    )

    # Initialize controller
    controller = GoalNavigationController()

    # Run episode
    obs, info = env.reset()
    obs_history = {}

    # Get goal position from environment
    goal_position = np.array(env.unwrapped.config["aircraft_config"][0]["task_config"]["config"]["position"])
    print(f"Goal position: {goal_position}")

    for _ in range(2000):  # Longer episode for goal navigation
        # Get observation dict
        obs_dict = env.unwrapped._observation.to_dict(obs)

        # Get control action
        act_dict = controller(obs_dict, goal_position)
        act = env.unwrapped._action.from_dict(act_dict)

        # Step environment
        obs, reward, truncated, terminated, info = env.step(act)

        # Update history
        obs_history = update_history(obs_history, obs_dict, act_dict, reward)

        if terminated or truncated:
            print("Episode finished:", "Success" if terminated else "Time limit")
            break

    # Plot results
    plot_goal_navigation(obs_history, goal_position)

if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    main()
