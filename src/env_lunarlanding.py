import pandas as pd
import numpy as np


class EnvironmentLunarLanding:
    def __init__(self, default_reward=0):
        # initialize instance variables

        # landing zone
        self.landing_zone = (50, 0)

        # start state (velocity_x, velocity_y, angle, position_x, position_y, landing_zone_x, landing_zone_y, fuel)
        self.start_state = (0, -10, 0, 25, 100, self.landing_zone[0], self.landing_zone[1], 25)

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

        vel_x, vel_y, angle, pos_x, pos_y, land_x, land_y, fuel = state

        # get new state
        if action == 'main_thruster':
            angle = angle % 360
            vel_x += 2 * np.sin(np.deg2rad(angle))
            vel_y += 2 * np.cos(np.deg2rad(angle))
            fuel = np.max([0, fuel - 2])

        elif action == 'left_thruster':
            angle += 10
            angle = angle % 360
            vel_x += 1 * np.sin(np.deg2rad(angle))
            vel_y += 1 * np.cos(np.deg2rad(angle))
            fuel = np.max([0, fuel - 1])

        elif action == 'right_thruster':
            angle -= 10
            angle = angle % 360
            vel_x += 1 * np.sin(np.deg2rad(angle))
            vel_y += 1 * np.cos(np.deg2rad(angle))
            fuel = np.max([0, fuel - 1])

        elif action == 'no_thruster':
            angle = angle % 360
            vel_x = vel_x
            vel_y = np.max([vel_y - 1, -10])
            fuel = fuel

        pos_x = pos_x + vel_x
        pos_y = np.max([0, pos_y + vel_y])

        terminal = False
        reward = self.default_reward
        state = (vel_x, vel_y, angle, pos_x, pos_y, self.landing_zone[0], self.landing_zone[1], fuel)

        # determine reward and if the agent is in a terminal state.
        if pos_y == 0 or fuel == 0:
            terminal = True

        # calculate reward when landing
        if fuel == 0:
            reward = -10000
        elif pos_y == 0:
            reward += -1 * np.sqrt((vel_x**2 + vel_y**2))

            if 5 <= angle <= 180:
                reward += -np.abs(angle - 5)
            elif 180 < angle <= 355:
                reward += -np.abs(355 - angle)

            reward += -np.abs(pos_x - self.landing_zone[0])**2

            reward += np.sqrt(fuel)

        reward = np.max([-10000, reward])

        return reward, state, terminal
