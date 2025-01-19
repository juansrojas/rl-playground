import pandas as pd
import numpy as np


class EnvironmentMountainCar:
    def __init__(self, default_reward=-1):
        # initialize instance variables
        self.min_position = -1.2
        self.max_position = 0.6

        self.min_max_velocity = 0.07

        # start state (position, velocity)
        self.start_state = (-0.5, 0)

        # goal state (position, velocity)
        self.goal_state = (0.5, 0)

        # physics params
        self.force = 0.001
        self.gravity = 0.0025

        # default reward
        self.default_reward = default_reward

    def env_start(self, start_state=None):
        # return initial state
        if pd.isnull(start_state):
            return self.start_state
        else:
            return start_state

    def env_step(self, state, action):
        # return new state and reward

        # get agent's current location
        position, velocity = state

        if action == 'accelerate_left':
            action_effect = -1

        elif action == 'dont_accelerate':
            action_effect = 0

        elif action == 'accelerate_right':
            action_effect = 1

        else:
            raise ValueError("Wrong action value")

        velocity += action_effect * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.min_max_velocity, self.min_max_velocity)

        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if position == self.min_position and velocity < 0:
            velocity = 0

        terminal = bool(position >= self.goal_state[0] and velocity >= self.goal_state[1])

        if terminal:
            reward = 0
        else:
            reward = self.default_reward

        return reward, (position, velocity), terminal
