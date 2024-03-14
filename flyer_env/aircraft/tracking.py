import numpy as np
from typing import Dict

from flyer_env.utils import Vector


class TrackPoints:

    def __init__(
        self,
        goal_set: Dict[float, Vector],
        radius: float = 500.0,
    ):

        self.goal_state = 0
        self.control_state = 0

        self.goal_set = goal_set
        self.radius = radius

    @staticmethod
    def _bearing(a, b):
        point_diff = np.subtract(b, a)
        return np.arctan2(point_diff[1], point_diff[0])

    @staticmethod
    def _distance(a, b):
        point_diff = np.subtract(b, a)
        return np.linalg.norm(point_diff)

    @staticmethod
    def _check_angle(bearing):
        while bearing < 0.0:
            bearing += 2.0 * np.pi
        while bearing > 2.0 * np.pi:
            bearing -= 2.0 * np.pi
        return bearing

    @staticmethod
    def _check_angle_negative_range(bearing):
        while bearing > np.pi:
            bearing -= 2.0 * np.pi
        while bearing < -np.pi:
            bearing += 2.0 * np.pi
        return bearing

    @staticmethod
    def _unit_dir_vector(start_point, end_point):
        direction_vector = np.subtract(end_point, start_point)
        try:
            unit_vector_n = direction_vector[0] / np.sqrt(
                np.power(direction_vector[0], 2) + np.power(direction_vector[1], 2)
            )
        except ZeroDivisionError:
            unit_vector_n = 0
        try:
            unit_vector_e = direction_vector[1] / np.sqrt(
                np.power(direction_vector[0], 2) + np.power(direction_vector[1], 2)
            )
        except ZeroDivisionError:
            unit_vector_e = 0
        unit_vector = (unit_vector_n, unit_vector_e)
        return unit_vector

    def arc_path(self, pos):

        if self.goal_state < len(self.goal_set) - 2:

            # Get points defining trajecotries
            point_a = self.goal_set[self.goal_state]
            point_b = self.goal_set[self.goal_state + 1]
            point_c = self.goal_set[self.goal_state + 2]

            track_bearing_in = self._check_angle_negative_range(
                self._bearing(point_a, point_b)
            )
            track_bearing_out = self._check_angle_negative_range(
                self._bearing(point_b, point_c)
            )

            filet_angle = self._check_angle_negative_range(
                np.pi - (track_bearing_out - track_bearing_in)
            )
            if self.control_state == 0:
                q = self._unit_dir_vector(point_a, point_b)
                w = point_b
                try:
                    z_point = (
                        w[0] - (np.abs((self.radius / np.tan(filet_angle / 2))) * q[0]),
                        w[1] - (np.abs((self.radius / np.tan(filet_angle / 2))) * q[1]),
                    )
                    h_point = np.subtract(z_point, pos[:2])
                    h_val = (h_point[0] * q[0]) + (h_point[1] * q[1])
                    if h_val < 0:
                        # Entered h-plane transition to curved segment
                        self.control_state = 1

                    objective_bearing = self._bearing(pos[:2], point_b)
                    objective_distance = self._distance(pos[:2], point_b)
                    track_distance = self._distance(point_a, point_b)
                    off_track_angle = self._check_angle_negative_range(
                        objective_bearing - track_bearing_in
                    )
                    heading = (
                        0.5 * track_distance / objective_distance * off_track_angle
                    ) + track_bearing_in
                except ZeroDivisionError:
                    print(f"Straight lines between: {point_a}, {point_b}, {point_c}")

            if self.control_state == 1:
                q0 = self._unit_dir_vector(point_a, point_b)
                q1 = self._unit_dir_vector(point_b, point_c)
                q_grad = self._unit_dir_vector(q0, q1)
                w = point_b
                center_point = (
                    w[0]
                    + (np.abs((self.radius / np.sin(filet_angle / 2))) * q_grad[0]),
                    w[1]
                    + (np.abs((self.radius / np.sin(filet_angle / 2))) * q_grad[1]),
                )
                z_point = (
                    w[0] + (np.abs((self.radius / np.tan(filet_angle / 2))) * q1[0]),
                    w[1] + (np.abs((self.radius / np.tan(filet_angle / 2))) * q1[1]),
                )
                turning_direction = np.copysign(1, (q0[0] * q1[1]) - (q0[1] * q1[0]))
                h_point = np.subtract(z_point, pos[:2])
                h_val = (h_point[0] * q1[0]) + (h_point[1] * q1[1])
                if h_val < 0:
                    self.control_state = 0
                    self.goal_state += 1
                distance_from_center = np.sqrt(
                    np.power(pos[0] - center_point[0], 2)
                    + np.power(pos[1] - center_point[1], 2)
                )

                circ_x = center_point[0] - pos[0]
                circ_y = center_point[1] - pos[1]
                circle_angle = np.arctan2(circ_y, circ_x)
                if circle_angle < 0:
                    circle_angle = circle_angle + (2 * np.pi)
                tangent_track = circle_angle - (turning_direction * (np.pi / 2))
                if tangent_track < 0:
                    tangent_track = tangent_track + (2 * np.pi)
                if tangent_track > 2 * np.pi:
                    tangent_track = tangent_track - (2 * np.pi)
                error = (distance_from_center - self.radius) / self.radius
                k_orbit = 4.0
                heading = tangent_track + (np.arctan(k_orbit * error))
                # heading = tangent_track
        else:
            points = list(self.goal_set.values())

            point_a = points[-2]
            point_b = points[-1]

            track_bearing = self._check_angle_negative_range(
                self._bearing(point_a, point_b)
            )
            track_distance = self._distance(point_a, point_b)

            objective_bearing = self._bearing(pos[:2], point_b)
            objective_distance = self._distance(pos[:2], point_b)

            off_track_angle = self._check_angle_negative_range(
                objective_bearing - track_bearing
            )
            heading = (
                0.5 * (track_distance / objective_distance) * off_track_angle
            ) + track_bearing
            heading = self._check_angle_negative_range(heading)
            # print(f'heading: {heading * 180.0/np.pi}')
        return heading
