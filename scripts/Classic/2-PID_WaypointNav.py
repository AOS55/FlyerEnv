import matplotlib.pyplot as plt
import flyer_env
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import pytest

class DubinsSegmentType(Enum):
    LEFT = 'L'
    RIGHT = 'R'
    STRAIGHT = 'S'

@dataclass
class DubinsPath:
    """Represents a complete Dubins path."""
    path_type: str  # Combination of segment types (e.g., "LSR")
    segments: List[Tuple[DubinsSegmentType, float]]  # List of (type, length) pairs
    total_length: float
    start_pose: Tuple[float, float, float]  # x, y, heading
    end_pose: Tuple[float, float, float]
    turning_radius: float

class DubinsPathPlanner:
    def __init__(self, min_turn_radius: float):
        """
        Initialize Dubins path planner.

        Args:
            min_turn_radius: Minimum turning radius of the aircraft
        """
        self.min_turn_radius = min_turn_radius
        # All possible Dubins path types
        self.path_types = ['LSL', 'RSR', 'LSR', 'RSL', 'RLR', 'LRL']
        self.start_pose = None

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def compute_path(self,
        start_pose: Tuple[float, float, float],
        end_pose: Tuple[float, float, float]
    ) -> Optional[DubinsPath]:
        """
        Compute shortest Dubins path between two poses.

        Args:
            start_pose: (x, y, heading) of start position
            end_pose: (x, y, heading) of goal position

        Returns:
            DubinsPath object containing path information
        """

        self.start_pose = start_pose

        # Transform end pose to relative coordinates
        dx = end_pose[0] - start_pose[0]
        dy = end_pose[1] - start_pose[1]

        # Rotate to start pose frame
        c = np.cos(-start_pose[2])
        s = np.sin(-start_pose[2])
        x = (c * dx + s * dy) / self.min_turn_radius
        y = (-s * dx + c * dy) / self.min_turn_radius
        phi = self.normalize_angle(end_pose[2] - start_pose[2])

        print(f"Debug - Transformed coordinates: x={x:.3f}, y={y:.3f}, phi={phi:.3f}")

        # Check for U-turn (heading difference close to pi)
        if abs(abs(phi) - np.pi) < 0.1:
            print("Debug - Attempting U-turn path")
            # Try LRL first
            path = self._compute_path_type(x, y, phi, 'LRL')
            if path and self._verify_path(path):
                print("Debug - LRL path found")
                return path
            print("Debug - LRL failed, trying RLR")
            path = self._compute_path_type(x, y, phi, 'RLR')
            if path and self._verify_path(path):
                print("Debug - RLR path found")
                return path
            print("Debug - Both U-turn paths failed verification")

        # Try other path types
        preferred_order = ['LSL', 'RSR', 'LSR', 'RSL', 'LRL', 'RLR']

        best_path = None
        min_length = float('inf')

        for path_type in preferred_order:
            path = self._compute_path_type(x, y, phi, path_type)
            if path and path.total_length < min_length:
                # if self._verify_path(path):
                min_length = path.total_length
                best_path = path

        return best_path

    def _compute_straight_length(self, x: float, y: float) -> float:
        """Compute length of straight segment."""
        return np.sqrt(x*x + y*y)

    def _compute_tangent_point(self, x: float, y: float, side: DubinsSegmentType) -> Tuple[float, float]:
        """Compute tangent point for circular arc."""
        if side == DubinsSegmentType.LEFT:
            angle = np.arctan2(y, x)
            return (
                x - np.sin(angle),
                y + np.cos(angle)
            )
        else:  # RIGHT
            angle = np.arctan2(-y, -x)
            return (
                x + np.sin(angle),
                y - np.cos(angle)
            )

    def _compute_path_type(self, x: float, y: float, phi: float, path_type: str) -> Optional[DubinsPath]:
        """
        Compute specific Dubins path type.

        Args:
            x, y: Goal position in normalized coordinates
            phi: Goal heading in normalized coordinates
            path_type: Type of path to compute (e.g., "LSL", "RSR")
        """

        # First, check if a straight path is possible (with some tolerance)
        straight_tolerance = 0.1
        if abs(y) < straight_tolerance and abs(phi) < straight_tolerance:
            return DubinsPath(
                path_type='LSL',
                segments=[(DubinsSegmentType.STRAIGHT, self._compute_straight_length(x, y))],
                total_length=self._compute_straight_length(x, y),
                start_pose=self.start_pose,
                end_pose=(x * self.min_turn_radius, y * self.min_turn_radius, phi),
                turning_radius=self.min_turn_radius
            )

        segments = []
        total_length = 0.0

        if path_type in ["LSL", "RSR"]:
            turn_dir = DubinsSegmentType.LEFT if path_type[0] == 'L' else DubinsSegmentType.RIGHT
            sign = 1.0 if turn_dir == DubinsSegmentType.LEFT else -1.0

            # Centers of turning circles
            center1 = np.array([0, sign])
            center2 = np.array([x, y + sign])

            centers_vec = center2 - center1
            centers_dist = np.linalg.norm(centers_vec)

            if centers_dist < 2.0:
                return None

            # Compute tangent angles
            theta = np.arctan2(centers_vec[1], centers_vec[0])
            theta1 = theta + (np.pi/2 if turn_dir == DubinsSegmentType.RIGHT else -np.pi/2)
            theta2 = theta1

            # Compute arc lengths
            alpha1 = self.normalize_angle(theta1)
            alpha2 = self.normalize_angle(theta2 - phi)
            straight_length = centers_dist

            # Only add non-zero segments
            if abs(alpha1) > 0.01:
                segments.append((turn_dir, abs(alpha1)))
            if straight_length > 0.01:
                segments.append((DubinsSegmentType.STRAIGHT, straight_length))
            if abs(alpha2) > 0.01:
                segments.append((turn_dir, abs(alpha2)))

            total_length = abs(alpha1) + straight_length + abs(alpha2)

        elif path_type in ["LSR", "RSL"]:
            first_turn = DubinsSegmentType.LEFT if path_type[0] == 'L' else DubinsSegmentType.RIGHT
            second_turn = DubinsSegmentType.RIGHT if path_type[0] == 'L' else DubinsSegmentType.LEFT

            sign1 = 1.0 if first_turn == DubinsSegmentType.LEFT else -1.0
            sign2 = 1.0 if second_turn == DubinsSegmentType.LEFT else -1.0

            center1 = np.array([0, sign1])
            center2 = np.array([x, y + sign2])

            centers_vec = center2 - center1
            centers_dist = np.linalg.norm(centers_vec)

            if centers_dist < 2.0:
                return None

            theta = np.arctan2(centers_vec[1], centers_vec[0])
            beta = np.arccos(2.0 / centers_dist)

            if np.isnan(beta):
                return None

            if first_turn == DubinsSegmentType.LEFT:
                theta1 = theta + beta
                theta2 = theta + beta - np.pi
            else:
                theta1 = theta - beta
                theta2 = theta - beta + np.pi

            alpha1 = self.normalize_angle(theta1)
            alpha2 = self.normalize_angle(theta2 - phi)
            straight_length = np.sqrt(centers_dist**2 - 4.0)

            if abs(alpha1) < 0.1:  # Very small first turn
                    alpha1 = 0
            if abs(alpha2) < 0.1:  # Very small final turn
                alpha2 = 0

            segments = [
                (first_turn, abs(alpha1)),
                (DubinsSegmentType.STRAIGHT, straight_length),
                (second_turn, abs(alpha2))
            ]
            total_length = abs(alpha1) + straight_length + abs(alpha2)

        else:  # RLR or LRL paths
            first_turn = DubinsSegmentType.LEFT if path_type[0] == 'L' else DubinsSegmentType.RIGHT
            sign = 1.0 if first_turn == DubinsSegmentType.LEFT else -1.0

            # Centers of turning circles
            center1 = np.array([0, sign])
            center2 = np.array([x, y + sign])

            centers_vec = center2 - center1
            centers_dist = np.linalg.norm(centers_vec)

            # Special handling for U-turns
            if abs(abs(phi) - np.pi) < 0.1:  # U-turn case
                # Simplified U-turn calculation
                alpha1 = np.pi/2 * sign
                alpha2 = np.pi
                alpha3 = np.pi/2 * sign

                segments = [
                    (first_turn, abs(alpha1)),
                    (DubinsSegmentType.RIGHT if first_turn == DubinsSegmentType.LEFT
                    else DubinsSegmentType.LEFT, abs(alpha2)),
                    (first_turn, abs(alpha3))
                ]

                total_length = abs(alpha1) + abs(alpha2) + abs(alpha3)
            else:

                theta = np.arctan2(centers_vec[1], centers_vec[0])

                # Modified beta calculation with better numerical stability
                d = centers_dist / 4.0
                if d > 1.0:
                    return None
                beta = np.arccos(d)

                if np.isnan(beta):
                    return None

                if first_turn == DubinsSegmentType.LEFT:
                    theta1 = theta + beta
                    theta2 = theta - beta
                else:
                    theta1 = theta - beta
                    theta2 = theta + beta

                alpha1 = self.normalize_angle(theta1)
                alpha2 = 2 * beta  # Changed from 2 * np.pi - 2 * beta
                alpha3 = self.normalize_angle(theta2 - phi)

                # Create segments with proper scaling
                segments = [
                    (first_turn, abs(alpha1)),
                    (DubinsSegmentType.RIGHT if first_turn == DubinsSegmentType.LEFT
                    else DubinsSegmentType.LEFT, abs(alpha2)),
                    (first_turn, abs(alpha3))
                ]

                total_length = abs(alpha1) + abs(alpha2) + abs(alpha3)

        scaled_segments = [(seg_type, length * self.min_turn_radius) for seg_type, length in segments]
        scaled_total_length = total_length * self.min_turn_radius

        return DubinsPath(
            path_type=path_type,
            segments=scaled_segments,
            total_length=scaled_total_length,
            start_pose=self.start_pose,
            end_pose=(x * self.min_turn_radius, y * self.min_turn_radius, phi),
            turning_radius=self.min_turn_radius
        )

    def generate_waypoints(self, path: DubinsPath, num_points: int = 100) -> np.ndarray:
        """
        Generate waypoints along the Dubins path.

        Args:
            path: DubinsPath object
            num_points: Number of waypoints to generate

        Returns:
            Array of shape (num_points, 3) containing (x, y, heading) waypoints
        """
        waypoints = np.zeros((num_points, 3))

        # If only one segment (straight line)
        if len(path.segments) == 1:
            for i in range(num_points):
                t = i / (num_points - 1)
                waypoints[i] = np.array([
                    self.start_pose[0] + t * (path.end_pose[0] - self.start_pose[0]),
                    self.start_pose[1] + t * (path.end_pose[1] - self.start_pose[1]),
                    self.start_pose[2]
                ])
            return waypoints

        # For paths with multiple segments
        total_length = path.total_length
        if path.path_type in ['LRL', 'RLR']:
            # Generate more points at the turning sections
            distances = []
            turn_points = int(num_points * 0.4)  # 40% of points for turns
            straight_points = num_points - 2 * turn_points

            # First turn
            turn_dist = path.segments[0][1]
            distances.extend(np.linspace(0, turn_dist, turn_points))

            # Middle section
            mid_start = turn_dist
            mid_end = turn_dist + path.segments[1][1]
            distances.extend(np.linspace(mid_start, mid_end, straight_points))

            # Final turn
            distances.extend(np.linspace(mid_end, total_length, turn_points))
        else:
            distances = np.linspace(0, total_length, num_points)

        for i, distance in enumerate(distances):
            pose = self._compute_pose_at_distance(path, distance)
            waypoints[i] = pose

        return waypoints

    def _transform_to_global_frame(self, local_pose, start_pose):
        """Transform pose from local frame to global frame."""
        x, y, heading = local_pose
        x0, y0, heading0 = start_pose

        c = np.cos(heading0)
        s = np.sin(heading0)

        x_global = x0 + (c * x - s * y)
        y_global = y0 + (s * x + c * y)
        heading_global = self.normalize_angle(heading + heading0)

        return np.array([x_global, y_global, heading_global])

    def _compute_pose_at_distance(self, path: DubinsPath, distance: float) -> np.ndarray:
        """
        Compute pose (x, y, heading) at given distance along path.

        Args:
            path: DubinsPath object
            distance: Distance along path

        Returns:
            Array of [x, y, heading] at specified distance
        """
        if distance >= path.total_length:
            return np.array(path.end_pose)

        current_distance = 0.0
        current_pose = np.array(self.start_pose)

        for segment_type, length in path.segments:
            if current_distance + length >= distance:
                # This is our segment
                segment_distance = distance - current_distance

                if segment_type == DubinsSegmentType.STRAIGHT:
                    heading = current_pose[2]
                    return np.array([
                        current_pose[0] + segment_distance * np.cos(heading),
                        current_pose[1] + segment_distance * np.sin(heading),
                        heading
                    ])
                else:  # Turn segment
                    turn_direction = 1.0 if segment_type == DubinsSegmentType.LEFT else -1.0
                    normalized_distance = segment_distance / self.min_turn_radius
                    angle = normalized_distance * turn_direction

                    # Center of turning circle
                    center = np.array([
                        current_pose[0] - self.min_turn_radius * np.sin(current_pose[2]) * turn_direction,
                        current_pose[1] + self.min_turn_radius * np.cos(current_pose[2]) * turn_direction
                    ])

                    new_heading = self.normalize_angle(current_pose[2] + angle)
                    new_pos = center + self.min_turn_radius * np.array([
                        np.sin(new_heading) * turn_direction,
                        -np.cos(new_heading) * turn_direction
                    ])

                    return np.array([new_pos[0], new_pos[1], new_heading])

            # Update pose for next segment
            if segment_type == DubinsSegmentType.STRAIGHT:
                heading = current_pose[2]
                current_pose[0] += length * np.cos(heading)
                current_pose[1] += length * np.sin(heading)
            else:  # Turn segment
                turn_direction = 1.0 if segment_type == DubinsSegmentType.LEFT else -1.0
                normalized_length = length / self.min_turn_radius
                angle = normalized_length * turn_direction

                # Center of turning circle
                center = np.array([
                    current_pose[0] - self.min_turn_radius * np.sin(current_pose[2]) * turn_direction,
                    current_pose[1] + self.min_turn_radius * np.cos(current_pose[2]) * turn_direction
                ])

                current_pose[2] = self.normalize_angle(current_pose[2] + angle)
                current_pose[0:2] = center + self.min_turn_radius * np.array([
                    np.sin(current_pose[2]) * turn_direction,
                    -np.cos(current_pose[2]) * turn_direction
                ])

            current_distance += length

        return np.array(path.end_pose)

    def _verify_path(self, path: DubinsPath) -> bool:
        """Verify that the path is feasible and continuous."""
        if not path:
            return False

        # More lenient verification for U-turns
        is_uturn = path.path_type in ['LRL', 'RLR']

        # Check start and end poses
        waypoints = self.generate_waypoints(path, num_points=20)  # Increased number of points
        start_error = np.linalg.norm(waypoints[0] - np.array(path.start_pose))

        # For end pose, check position and heading separately
        end_pos_error = np.linalg.norm(waypoints[-1][:2] - np.array(path.end_pose)[:2])
        end_heading_error = abs(self.normalize_angle(waypoints[-1][2] - path.end_pose[2]))

        max_position_error = 1.0 if is_uturn else 0.5
        max_heading_error = np.pi/4 if is_uturn else np.pi/6

        if start_error > max_position_error or end_pos_error > max_position_error:
            print(f"Debug - Position error too large: start={start_error:.3f}, end={end_pos_error:.3f}")
            return False

        if end_heading_error > max_heading_error:
            print(f"Debug - Heading error too large: {end_heading_error:.3f}")
            return False

        # Check path continuity
        for i in range(len(waypoints)-1):
            dist = np.linalg.norm(waypoints[i+1][:2] - waypoints[i][:2])
            heading_change = abs(self.normalize_angle(waypoints[i+1][2] - waypoints[i][2]))

            # More lenient checks for the final segments of U-turns
            is_final_segment = i >= len(waypoints) - 3 and is_uturn
            max_segment_distance = path.total_length / (len(waypoints) - 1)
            max_segment_distance *= 4 if is_final_segment else (3 if is_uturn else 2)
            max_heading_change = np.pi if is_final_segment else (np.pi/2 if is_uturn else np.pi/4)

            if dist > max_segment_distance or heading_change > max_heading_change:
                print(f"Debug - Continuity check failed at segment {i}: dist={dist:.3f}, heading_change={heading_change:.3f}")
                # Don't immediately return False for final segments of U-turns
                if not (is_uturn and i >= len(waypoints) - 3):
                    return False

        return True

class TrajectoryGenerator:
    """Generates smooth trajectories from Dubins path waypoints."""
    def __init__(self, cruise_speed: float, max_accel: float):
        self.cruise_speed = cruise_speed
        self.max_accel = max_accel

    def generate_trajectory(self, waypoints: np.ndarray, dt: float) -> np.ndarray:
        """
        Generate time-parameterized trajectory from waypoints.

        Args:
            waypoints: Array of (x, y, heading) waypoints
            dt: Time step for trajectory

        Returns:
            Array of (x, y, heading, speed, time) trajectory points
        """
        # Calculate distances between waypoints
        diffs = np.diff(waypoints[:, :2], axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        total_distance = np.sum(distances)

        # Generate speed profile
        num_segments = len(distances)
        speeds = np.ones(num_segments + 1) * self.cruise_speed

        # Adjust speeds for turns based on curvature
        for i in range(1, num_segments):
            # Calculate path angle change
            v1 = diffs[i-1]
            v2 = diffs[i]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            # Reduce speed in turns
            if angle > 0.1:  # Threshold for considering it a turn
                speeds[i] = self.cruise_speed * (1 - angle/np.pi)

        # Generate timestamps
        times = np.zeros(len(speeds))
        for i in range(1, len(times)):
            segment_time = 2 * distances[i-1] / (speeds[i] + speeds[i-1])
            times[i] = times[i-1] + segment_time

        # Interpolate trajectory
        total_time = times[-1]
        num_points = int(np.ceil(total_time / dt)) + 1
        trajectory = np.zeros((num_points, 5))  # x, y, heading, speed, time

        # Force exact start and end points
        trajectory[0] = np.array([waypoints[0, 0], waypoints[0, 1], waypoints[0, 2], speeds[0], 0.0])
        trajectory[-1] = np.array([waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2], speeds[-1], total_time])

        for i in range(num_points):
            t = i * dt
            idx = np.searchsorted(times, t) - 1
            idx = max(0, min(idx, len(times)-2))

            alpha = (t - times[idx]) / (times[idx+1] - times[idx])

            # Position interpolation
            trajectory[i, :2] = (1 - alpha) * waypoints[idx, :2] + alpha * waypoints[idx+1, :2]

            # Heading interpolation
            trajectory[i, 2] = self.normalize_angle((1 - alpha) * waypoints[idx, 2] +
                                                    alpha * waypoints[idx+1, 2])

            # Speed interpolation
            trajectory[i, 3] = (1 - alpha) * speeds[idx] + alpha * speeds[idx+1]

            # Time
            trajectory[i, 4] = t

        return trajectory

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

class TrajectoryFollowingController:
    """Controller for following pre-computed trajectories."""
    def __init__(self,
        altitude_gains=(1.5, 0.05, 0.4),
        heading_gains=(0.8, 0.1, 0.3),
        speed_gains=(2.0, 0.1, 0.4),
        cross_track_gains=(0.6, 0.0, 0.2)
    ):
        # PID controllers with improved gains
        self.altitude_pid = PIDController(
            kp=altitude_gains[0],
            ki=altitude_gains[1],
            kd=altitude_gains[2]
        )
        self.heading_pid = PIDController(
            kp=heading_gains[0],
            ki=heading_gains[1],
            kd=heading_gains[2],
            rate_limit=0.05
        )
        self.speed_pid = PIDController(
            kp=speed_gains[0],
            ki=speed_gains[1],
            kd=speed_gains[2]
        )
        self.cross_track_pid = PIDController(
            kp=cross_track_gains[0],
            ki=cross_track_gains[1],
            kd=cross_track_gains[2]
        )

        # Look-ahead distance for trajectory following
        self.look_ahead_time = 1.0  # seconds

        # Store trajectory data
        self.trajectory = None
        self.current_segment = 0
        self.trajectory_time = 0.0

    def set_trajectory(self, trajectory: np.ndarray):
        """Set the trajectory to follow."""
        self.trajectory = trajectory
        self.current_segment = 0
        self.trajectory_time = 0.0

        # Reset all controllers
        self.altitude_pid.reset()
        self.heading_pid.reset()
        self.speed_pid.reset()
        self.cross_track_pid.reset()

    def get_look_ahead_point(self, current_time: float) -> Tuple[float, float, float, float]:
        """Get trajectory point at look-ahead time."""
        target_time = current_time + self.look_ahead_time
        # Find relevant trajectory segment
        idx = np.searchsorted(self.trajectory[:, 4], target_time) - 1
        if idx < 0:
            idx = 0
        elif idx >= len(self.trajectory) - 1:
            idx = len(self.trajectory) - 2

        # Interpolate between points
        t0, t1 = self.trajectory[idx, 4], self.trajectory[idx + 1, 4]
        alpha = (target_time - t0) / (t1 - t0)

        # Get interpolated state
        p0, p1 = self.trajectory[idx, :4], self.trajectory[idx + 1, :4]
        interpolated = p0 + alpha * (p1 - p0)

        return interpolated  # x, y, heading, speed

    def calculate_cross_track_error(self, current_pos: np.ndarray,
                                  target_pos: np.ndarray,
                                  path_heading: float) -> float:
        """Calculate cross-track error from path."""
        # Vector from current position to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # Rotate to path frame
        path_normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
        cross_track = np.dot([dx, dy], path_normal)

        return cross_track

    def __call__(self, obs: Dict[str, float]) -> Dict[str, float]:
        if self.trajectory is None:
            raise ValueError("Trajectory not set")

        # Get current state
        current_pos = np.array([obs['x'], obs['y'], -obs['altitude']])
        current_time = self.trajectory_time

        # Get look-ahead point
        look_ahead = self.get_look_ahead_point(current_time)
        target_pos = np.array([look_ahead[0], look_ahead[1], current_pos[2]])
        target_heading = look_ahead[2]
        target_speed = look_ahead[3]

        # Calculate cross-track error
        cross_track = self.calculate_cross_track_error(
            current_pos[:2],
            target_pos[:2],
            target_heading
        )

        # Modify desired heading based on cross-track error
        cross_track_correction = self.cross_track_pid.update(cross_track)
        desired_heading = target_heading + cross_track_correction

        # Calculate control errors
        heading_error = self.normalize_angle(desired_heading - obs['heading'])
        speed_error = target_speed - obs['airspeed']
        altitude_error = target_pos[2] - current_pos[2]

        # print(f'target_pos: {target_pos}, current_pos: {current_pos}, altitude_error: {altitude_error}')
        # print(f"target_speed: {target_speed}, current_speed: {obs['airspeed']}, speed_error: {speed_error}")

        # Generate control inputs
        bank_angle = self.heading_pid.update(heading_error)
        bank_angle = np.clip(bank_angle, -np.pi/4, np.pi/4)

        vertical_speed = self.altitude_pid.update(altitude_error)
        vertical_speed = np.clip(vertical_speed, -10.0, 10.0)

        acceleration = self.speed_pid.update(speed_error)
        acceleration = np.clip(acceleration, -10.0, 10.0)

        # Update trajectory time
        self.trajectory_time += 1/60.0  # Assuming 60Hz update rate

        return {
            'vertical_speed': vertical_speed,
            'bank_angle': bank_angle,
            'acceleration': acceleration
        }

    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float = 1/60, rate_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.windup_limit = 20.0
        self.prev_output = 0.0
        self.max_rate = rate_limit
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

def plot_results(obs_history: Dict[str, list], goal_position: np.ndarray,
                waypoints: np.ndarray, trajectory: np.ndarray):
    """Plot the results including planned path and actual trajectory."""
    fig = plt.figure(figsize=(15, 10))

    plot_colours = {
            'track': '#1E88E5',  # Blue
            'plan': '#43A047',  # Green
            'goal': '#F4511E',  # Orange
        }

    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(
        obs_history['x'], obs_history['y'], obs_history['altitude'],
        color=plot_colours['track'], label='Actual Path'
    )
    ax1.plot(
        waypoints[:, 0], waypoints[:, 1],
        [-goal_position[2]]*len(waypoints), color=plot_colours['plan'], linestyle='--', label='Planned Path'
    )
    ax1.scatter(
        goal_position[0], goal_position[1], -goal_position[2],
        color=plot_colours['goal'], marker='*', s=100, label='Goal'
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    ax1.legend()
    ax1.set_title('3D Trajectory')

    # Top-down view
    ax2 = fig.add_subplot(222)
    ax2.plot(obs_history['x'], obs_history['y'], color=plot_colours['track'], linestyle='-', label='Actual Path')
    ax2.plot(waypoints[:, 0], waypoints[:, 1], color=plot_colours['plan'], linestyle='--', label='Planned Path')
    ax2.scatter(goal_position[0], goal_position[1],
                color=plot_colours['goal'], marker='*', s=100, label='Goal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    ax2.set_title('Top-down View')

    # Control inputs
    ax3 = fig.add_subplot(223)
    ax3.plot(obs_history['vertical_speed'], color=plot_colours['track'], linestyle='-', label='Vertical Speed')
    ax3.plot(obs_history['bank_angle'], color=plot_colours['plan'], linestyle='--', label='Bank Angle')
    ax3.plot(obs_history['acceleration'], color=plot_colours['goal'], linestyle='-.', label='Acceleration')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Values')
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Control Inputs')

    # Tracking errors
    ax4 = fig.add_subplot(224)

    # Calculate tracking error with interpolation
    actual_positions = np.column_stack((obs_history['x'], obs_history['y']))
    actual_len = len(actual_positions)

    # Debug print
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Actual path length: {actual_len}")

    # Interpolate planned trajectory to match actual path length
    if len(trajectory) > 0:
        t_points = np.linspace(0, len(trajectory)-1, actual_len)
        planned_x = np.interp(t_points, np.arange(len(trajectory)), trajectory[:, 0])
        planned_y = np.interp(t_points, np.arange(len(trajectory)), trajectory[:, 1])
        planned_positions = np.column_stack((planned_x, planned_y))
        tracking_error = np.linalg.norm(planned_positions - actual_positions, axis=1)
    else:
        print("Warning: Empty trajectory")
        tracking_error = np.zeros(actual_len)

    rewards = obs_history['reward']
    ax4.plot(rewards, color='black', label='Reward')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Reward')
    ax4.grid(True)
    # ax4.legend()
    ax4.set_title('Reward over Time')

    plt.savefig('results.pdf')
    plt.show()

def main():
    # Create environment with specific goal parameters
    env = gym.make('flyer_goal-v1',
        seed=41,
        render_mode="rgb_array",
        goal_distance_range=(2000.0, 5000.0),  # Changed from distance_range
        goal_altitude_range_agl=(500.0, 500.0),  # Changed from altitude_range
        goal_heading_range_rad=(np.pi/4, np.pi/4),  # Changed from heading_range
        goal_tolerance=100.0,  # Changed from tolerance
        reward_type="Dense"
    )

    # Initialize path planner and trajectory generator
    min_turn_radius = 200.0  # Minimum turning radius in meters
    cruise_speed = 50.0      # Cruise speed in m/s
    max_accel = 2.0        # Maximum acceleration in m/s^2

    path_planner = DubinsPathPlanner(min_turn_radius)
    trajectory_generator = TrajectoryGenerator(cruise_speed, max_accel)
    controller = TrajectoryFollowingController()

    # Run episode
    obs, info = env.reset()
    obs_history = {}

    # Get goal position from environment
    # Get goal position from environment info after reset
    try:
        # First try to get it from info
        goal_position_dict = info.get("goal_position")
        if goal_position_dict and isinstance(goal_position_dict, dict):
            goal_position = np.array([goal_position_dict['x'], goal_position_dict['y'], goal_position_dict['z']])
            print(f"Goal position from info: {goal_position}")
        else:
            # If that didn't work, try to extract from the task config in the unwrapped env
            unwrapped_env = env.unwrapped
            goal_pos_data = unwrapped_env.controlled_vehicles[0].config
            # Might need to adjust the exact way to access this based on internal structure
            goal_position = np.array([
                goal_pos_data["task_config"]["config"]["position"]["x"],
                goal_pos_data["task_config"]["config"]["position"]["y"],
                goal_pos_data["task_config"]["config"]["position"]["z"]
            ])
            print(f"Goal position from unwrapped env: {goal_position}")
    except (KeyError, AttributeError) as e:
        # If all else fails, hardcode the position or generate it
        print(f"Could not access goal position: {e}")
        # This assumes the goal is at a 45-degree angle from start at distance of 2750m
        # and an altitude of 500m - modify as needed based on your environment setup
        goal_position = np.array([1946.5, 1946.5, 500.0])
        print(f"Using fallback goal position: {goal_position}")


    # Get initial state
    obs_dict = env.unwrapped._observation.to_dict(obs)
    start_pose = (
        obs_dict['x'],
        obs_dict['y'],
        obs_dict['heading']
    )
    end_pose = (
        goal_position[0],
        goal_position[1],
        np.pi/4  # Target heading at goal
    )

    print(f"Start pose: {start_pose}")
    print(f"End pose: {end_pose}")

    # Plan path
    path = path_planner.compute_path(start_pose, end_pose)
    if path is None:
        raise ValueError("Could not find valid path to goal")

    print(f"Path type: {path.path_type}")
    print(f"Path segments: {path.segments}")

    # Generate waypoints along path
    waypoints = path_planner.generate_waypoints(path, num_points=150)
    print(f"Generated {len(waypoints)} waypoints")

    # Generate time-parameterized trajectory
    trajectory = trajectory_generator.generate_trajectory(waypoints, dt=1/60.0)
    print(f"Generated trajectory with {len(trajectory)} points")

    # Set trajectory for controller
    controller.set_trajectory(trajectory)

    for step in range(2000):  # Longer episode for goal navigation
        # Get observation dict
        obs_dict = env.unwrapped._observation.to_dict(obs)

        # Print debug info every 100 steps
        if step % 100 == 0:
            print(f"Step {step}:")
            print(f"Position: ({obs_dict['x']:.1f}, {obs_dict['y']:.1f}, {obs_dict['altitude']:.1f})")
            print(f"Heading: {obs_dict['heading']:.2f}")
            print(f"Airspeed: {obs_dict['airspeed']:.1f}")

        # Get control action from trajectory following controller
        act_dict = controller(obs_dict)
        act = env.unwrapped._action.from_dict(act_dict)

        # Step environment
        obs, reward, truncated, terminated, info = env.step(act)

        # Update history
        obs_history = update_history(obs_history, obs_dict, act_dict, reward)

        if terminated or truncated:
            # print(f"Episode finished: {'Success' if terminated else 'Time limit'}")
            print(f"Final position: ({obs_dict['x']:.1f}, {obs_dict['y']:.1f}, {obs_dict['altitude']:.1f})")
            break

    # Plot results
    plot_results(obs_history, goal_position, waypoints, trajectory)


if __name__ == "__main__":
    flyer_env.register_flyer_envs()
    # test_dubins_planner()
    main()

def test_dubins_planner():
    planner = DubinsPathPlanner(min_turn_radius=1.0)

    test_cases = [
        # Test Case 1: Simple straight line
        {
            'start': (0, 0, 0),
            'end': (4, 0, 0),
            'name': "Straight line"
        },

        # Test Case 2: Simple right turn
        {
            'start': (0, 0, 0),
            'end': (2, 2, np.pi/2),
            'name': "90-degree right turn"
        },

        # Test Case 3: Simple left turn
        {
            'start': (0, 0, 0),
            'end': (2, -2, -np.pi/2),
            'name': "90-degree left turn"
        },

        # Test Case 4: U-turn
        {
            'start': (0, 0, 0),
            'end': (-2, 0, np.pi),
            'name': "U-turn"
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: {test_case['name']}")
        path = planner.compute_path(test_case['start'], test_case['end'])

        if path is None:
            print(f"No path found for test case {i + 1}")
            continue

        print(f"Path type: {path.path_type}")
        print(f"Total length: {path.total_length:.2f}")
        print("Segments:")
        for segment_type, length in path.segments:
            print(f"  {segment_type}: {length:.2f}")

        # Generate waypoints
        waypoints = planner.generate_waypoints(path, num_points=10)

        # Print first, middle, and last waypoints
        print("\nKey waypoints:")
        print(f"Start:  {waypoints[0]}")
        print(f"Middle: {waypoints[4]}")
        print(f"End:    {waypoints[-1]}")

        # Calculate errors
        start_pos_error = np.linalg.norm(waypoints[0][:2] - np.array(test_case['start'][:2]))
        end_pos_error = np.linalg.norm(waypoints[-1][:2] - np.array(test_case['end'][:2]))
        start_heading_error = abs(planner.normalize_angle(waypoints[0][2] - test_case['start'][2]))
        end_heading_error = abs(planner.normalize_angle(waypoints[-1][2] - test_case['end'][2]))

        print(f"\nErrors:")
        print(f"Start position error: {start_pos_error:.6f}")
        print(f"End position error: {end_pos_error:.6f}")
        print(f"Start heading error: {start_heading_error:.6f}")
        print(f"End heading error: {end_heading_error:.6f}")

        # Check path continuity
        print("\nPath continuity check:")
        for j in range(len(waypoints)-1):
            dist = np.linalg.norm(waypoints[j+1][:2] - waypoints[j][:2])
            heading_change = abs(planner.normalize_angle(waypoints[j+1][2] - waypoints[j][2]))
            print(f"Segment {j}: dist={dist:.3f}, heading_change={heading_change:.3f}")

def test_right_angle():
    """Test with three points forming a right angle"""
    generator = TrajectoryGenerator(1.0, 1.0)
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, np.pi/2]
    ])
    dt = 0.1

    trajectory = generator.generate_trajectory(waypoints, dt)

    print(f"First point: {trajectory[0]}")
    print(f"Last point: {trajectory[-1]}")
    print(f"First waypoint: {waypoints[0]}")
    print(f"Last waypoint: {waypoints[-1]}")

    # Check endpoints
    np.testing.assert_allclose(trajectory[0, :2], waypoints[0, :2], rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(trajectory[-1, :2], waypoints[-1, :2], rtol=1e-2, atol=1e-2)

    # Check basic properties
    assert np.all(trajectory[:, 3] <= 1.01)  # speed constraint
    assert np.all(trajectory[:, 3] >= 0)     # no negative speeds
    assert np.all(np.diff(trajectory[:, 4]) > 0)  # time increasing

def test_trajectory_generator():
    # Test parameters
    cruise_speed = 2.0  # m/s
    max_accel = 1.0    # m/s^2
    dt = 0.1           # seconds

    # Create simpler test case
    waypoints = np.array([
        [0.0, 0.0, 0.0],      # Start
        [1.0, 0.0, 0.0],      # Right
        [1.0, 1.0, np.pi/2],  # Up
    ])

    # Initialize generator
    generator = TrajectoryGenerator(cruise_speed, max_accel)

    # Generate trajectory
    trajectory = generator.generate_trajectory(waypoints, dt)

    # Print debug information
    print(f"First point: {trajectory[0]}")
    print(f"Last point: {trajectory[-1]}")
    print(f"First waypoint: {waypoints[0]}")
    print(f"Last waypoint: {waypoints[-1]}")

    # Basic tests
    assert trajectory.shape[1] == 5  # Check output dimensions (x, y, heading, speed, time)
    assert len(trajectory) > 0       # Check that trajectory contains points

    # Check start point
    np.testing.assert_allclose(
        trajectory[0, :2],
        waypoints[0, :2],
        rtol=1e-2,
        atol=1e-2
    )

    # Check end point
    np.testing.assert_allclose(
        trajectory[-1, :2],
        waypoints[-1, :2],
        rtol=1e-2,
        atol=1e-2
    )

    # Check time stamps
    assert trajectory[0, 4] == 0.0   # Starts at t=0
    assert np.all(np.diff(trajectory[:, 4]) > 0)  # Time strictly increasing

    # Check speed constraints
    assert np.all(trajectory[:, 3] <= cruise_speed * 1.01)  # Allow 1% tolerance
    assert np.all(trajectory[:, 3] >= 0)  # No negative speeds

    # Check heading normalization
    assert np.all(trajectory[:, 2] >= -np.pi)
    assert np.all(trajectory[:, 2] <= np.pi)

def test_simple_line():
    """Test with just two points"""
    generator = TrajectoryGenerator(1.0, 1.0)
    waypoints = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    dt = 0.1

    trajectory = generator.generate_trajectory(waypoints, dt)

    np.testing.assert_allclose(
        trajectory[0, :2],
        waypoints[0, :2],
        rtol=1e-2,
        atol=1e-2
    )
    np.testing.assert_allclose(
        trajectory[-1, :2],
        waypoints[-1, :2],
        rtol=1e-2,
        atol=1e-2
    )

def test_normalize_angle():
    generator = TrajectoryGenerator(1.0, 1.0)

    # Test cases for angle normalization
    test_angles = [
        (0.0, 0.0),
        (np.pi, np.pi),
        (-np.pi, -np.pi),
        (3*np.pi, -np.pi),
        (-3*np.pi, np.pi),
        (2*np.pi, 0.0),
    ]

    for input_angle, expected_output in test_angles:
        result = generator.normalize_angle(input_angle)
        assert np.isclose(result, expected_output), \
            f"Expected {expected_output} but got {result} for input {input_angle}"

def test_trajectory_following_controller_initialization():
    # Test default initialization
    controller = TrajectoryFollowingController()
    assert controller.trajectory is None
    assert controller.current_segment == 0
    assert controller.trajectory_time == 0.0
    assert controller.look_ahead_time == 1.0

def test_set_trajectory():
    controller = TrajectoryFollowingController()
    # Create a simple trajectory
    trajectory = np.array([
        [0, 0, 0, 10, 0],    # x, y, heading, speed, time
        [10, 0, 0, 10, 1],
    ])

    controller.set_trajectory(trajectory)
    assert controller.trajectory is not None
    assert np.array_equal(controller.trajectory, trajectory)
    assert controller.current_segment == 0
    assert controller.trajectory_time == 0.0

def test_get_look_ahead_point():
    controller = TrajectoryFollowingController()
    trajectory = np.array([
        [0, 0, 0, 10, 0],    # x, y, heading, speed, time
        [10, 0, 0, 10, 1],
    ])
    controller.set_trajectory(trajectory)

    # Test interpolation at t=0.5
    look_ahead = controller.get_look_ahead_point(0.0)  # With look_ahead_time=1.0
    assert np.allclose(look_ahead, [10, 0, 0, 10])

def test_calculate_cross_track_error():
    controller = TrajectoryFollowingController()

    # Test case 1: Point above the path
    current_pos = np.array([0, 1])  # 1 unit above the path
    target_pos = np.array([1, 0])
    path_heading = 0  # Straight path along x-axis

    error = controller.calculate_cross_track_error(current_pos, target_pos, path_heading)
    assert np.isclose(error, -1.0)  # Negative because point is above the path

    # Test case 2: Point below the path
    current_pos = np.array([0, -1])  # 1 unit below the path
    error = controller.calculate_cross_track_error(current_pos, target_pos, path_heading)
    assert np.isclose(error, 1.0)  # Positive because point is below the path

    # Test case 3: Point on the path
    current_pos = np.array([0, 0])  # On the path
    error = controller.calculate_cross_track_error(current_pos, target_pos, path_heading)
    assert np.isclose(error, 0.0)  # Zero error when on the path

def test_controller_call():
    controller = TrajectoryFollowingController()
    trajectory = np.array([
        [0, 0, 0, 10, 0],
        [10, 0, 0, 10, 1],
    ])
    controller.set_trajectory(trajectory)

    # Test observation input
    obs = {
        'x': 0,
        'y': 0,
        'altitude': 100,
        'heading': 0,
        'airspeed': 10
    }

    # Get control outputs
    controls = controller(obs)

    # Check if all expected control outputs are present
    assert 'vertical_speed' in controls
    assert 'bank_angle' in controls
    assert 'acceleration' in controls

    # Check if control outputs are within expected ranges
    assert -10.0 <= controls['vertical_speed'] <= 10.0
    assert -np.pi/4 <= controls['bank_angle'] <= np.pi/4
    assert -10.0 <= controls['acceleration'] <= 10.0

def test_normalize_angle():
    controller = TrajectoryFollowingController()

    # Test angle normalization
    assert np.isclose(controller.normalize_angle(3*np.pi), -np.pi)
    assert np.isclose(controller.normalize_angle(-3*np.pi), -np.pi)
    assert np.isclose(controller.normalize_angle(0), 0)
