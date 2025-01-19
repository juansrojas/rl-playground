import pandas as pd
import numpy as np


class EnvironmentPendulum:
    def __init__(self):
        # initialize instance variables
        self.min_max_velocity = 2 * np.pi
        self.min_max_torque = 3
        self.dt = 0.05
        self.g = 9.8
        self.m = 1/3
        self.l = 3/2

        # start state (angle, angular velocity)
        self.start_state = (np.pi, 0.0)

    def env_start(self, start_state=None):
        # return initial state
        if pd.isnull(start_state):
            return self.start_state
        else:
            return start_state

    @staticmethod
    def angle_normalize(theta):
        return ((theta + np.pi) % (2 * np.pi)) - np.pi

    def env_step(self, state, action, terminal=False):
        theta, theta_dot = state

        if action == 'accelerate_left':
            torque = -self.min_max_torque

        elif action == 'dont_accelerate':
            torque = 0

        elif action == 'accelerate_right':
            torque = self.min_max_torque

        else:
            raise ValueError("Wrong action value")

        new_theta_dot = theta_dot + (3 * self.g / (2 * self.l) * np.sin(theta) + 3.0 / (self.m * self.l**2) * torque) * self.dt

        new_theta = theta + new_theta_dot * self.dt

        if new_theta_dot <= -self.min_max_velocity or new_theta_dot >= self.min_max_velocity:
            new_theta, new_theta_dot = self.start_state

        cost = -1 * self.angle_normalize(new_theta)**2

        return cost, (self.angle_normalize(new_theta), new_theta_dot), terminal
