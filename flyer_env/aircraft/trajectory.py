import numpy as np
from flyer_env.utils import Vector


class TrajectoryTarget:

    def __init__(self,
                 speed: float,
                 start_position: Vector,
                 start_heading: float = 0.0,
                 start_time: float = 0.0):
        """
        Target Object used to generate error for trajectory following objectives

        :param speed:
        :param start_position:
        :param start_time:
        """
        self.speed = speed
        self.position = start_position
        self.heading = start_heading
        self.time = start_time

    def update(self, traj_func, dt: float):
        self.time += dt
        self.position, done, self.heading = traj_func(self.position, self.time, dt, self.heading)
        return self.position.copy(), done

    def straight_and_level(self, length: float = 40.0, **kwargs):
        """
        Create an update function to maintain straight and level flight

        :param length: length of time to run for [s]
        :return: function used to update the targets position
        """
        def traj_func(position: Vector, time: float, dt: float, heading: float):
            position[0] += np.cos(heading) * self.speed * dt
            position[1] += np.sin(heading) * self.speed * dt
            position[2] += 0.0

            if time > length:
                done = True
            else:
                done = False

            return position, done
        return traj_func

    def climb(self, final_height: float = 200.0,
              climb_angle: float = 20.0 * np.pi/180.0, length: float = 15.0, **kwargs):
        """
        Create an update function to climb to an altitude

        :param final_height: final height gained [m]
        :param climb_angle: angle to climb at [rads]
        :param length: length of time to run for [s]
        :return: function used to update the targets position
        """
        start_height = self.position[2]

        def traj_func(position: Vector, time: float, dt: float, heading: float):
            height = position[2] - start_height
            # Straight and Level
            if time < 5.0:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Climb to height
            elif height < final_height:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += np.sin(climb_angle) * self.speed * dt
            # Straight and level
            else:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0

            if time > length:
                done = True
            else:
                done = False

            return position, done, heading
        return traj_func

    def descend(self, final_height: float = 200.0, climb_angle: float = -20.0 * np.pi/180.0,
                length: float = 15.0, **kwargs):
        """
        Create an update function to descend to an altitude

        :param final_height: final height lost [m]
        :param climb_angle: angle to climb at [rads]
        :param length: length of time to run for [s]
        :return: function used to update the targets position
        """
        start_height = self.position[2]

        def traj_func(position: Vector, time: float, dt: float, heading: float):
            height = position[2] - start_height
            # Straight and Level
            if time < 5.0:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Descend to height
            elif height > final_height:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += np.sin(climb_angle) * self.speed * dt
            # Straight and level
            else:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0

            if time > length:
                done = True
            else:
                done = False

            return position, done, heading
        return traj_func

    def left_turn(self, end_heading: float = -90.0*(np.pi/180.0),
                  turn_rate: float = 5.0*(np.pi/180.0), length: float = 45.0, **kwargs):
        """
        Create an update function to turn, left, to a new heading

        :param end_heading: final heading [rads]
        :param turn_rate: rate of turn [rads/s]
        :param length: length of time to run for [s]
        :return: function used to update the targets position
        """
        def traj_func(position: Vector, time: float, dt: float, heading: float):
            # Straight and Level
            if time < 5.0:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Left Turn
            elif np.isclose(heading, end_heading, rtol=1.0):
                heading -= turn_rate * dt
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Straight and Level
            else:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0

            if time > length:
                done = True
            else:
                done = False

            return position, done, heading
        return traj_func

    def right_turn(self, end_heading: float = 90.0*(np.pi/180.0),
                   turn_rate: float = 5.0*(np.pi/180.0), length: float = 45.0, **kwargs):
        """
        Create an update function to turn, right, to a new heading

        :param end_heading: final heading [rads]
        :param turn_rate: rate of turn [rads/s]
        :param length: length of time to run for [s]
        :return: function used to update the targets position
        """
        def traj_func(position: Vector, time: float, dt: float, heading: float):
            # Straight and Level
            if time < 5.0:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Left Turn
            elif np.isclose(heading, end_heading, rtol=1.0):
                heading += turn_rate * dt
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0
            # Straight and Level
            else:
                position[0] += np.cos(heading) * self.speed * dt
                position[1] += np.sin(heading) * self.speed * dt
                position[2] += 0.0

            if time > length:
                done = True
            else:
                done = False

            return position, done, heading
        return traj_func


if __name__ == "__main__":

    SPEED = 100.0
    START_POSITION = np.array([0.0, 0.0, 1000.0])
    dT = 1/60.0
    START_TIME = 0.0

    tt = TrajectoryTarget(SPEED, START_POSITION, START_TIME)
    poss = []
    times = []
    done = False
    # tt_desc = tt.descend(climb_angle=-5.0 * np.pi/180.0, length=100.0, final_height=-400.0)
    # tt_lt = tt.left_turn()
    tt_rt = tt.right_turn()
    while not done:
        pos, done = tt.update(tt_rt, dT)
        poss.append(pos)
        times.append(tt.time)

    poss = np.array(poss)
    import matplotlib.pyplot as plt

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(poss[:, 0], poss[:, 1], poss[:, 2])
    plt.plot(poss[:, 0], poss[:, 1])
    plt.show()
