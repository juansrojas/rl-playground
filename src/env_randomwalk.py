import numpy as np
import pandas as pd


class EnvironmentRandomWalk:
    def __init__(self, num_states=500, start_state=250, left_terminal_state=0, right_terminal_state=501, max_movement=100):
        # initialize instance variables
        self.num_states = num_states
        self.start_state = start_state
        self.left_terminal_state = left_terminal_state
        self.right_terminal_state = right_terminal_state
        self.max_movement = max_movement

    def env_start(self, start_state=None):
        # return first state from the environment
        if pd.isnull(start_state):
            return self.start_state
        else:
            return start_state

    def env_step(self, last_state, action):
        # return new state and reward
        if action == 'left':
            current_state = max(self.left_terminal_state, last_state + np.random.choice(range(-1*self.max_movement, 0)))

        elif action == 'right':
            current_state = min(self.right_terminal_state, last_state + np.random.choice(range(1, self.max_movement+1)))

        else:
            raise ValueError("Wrong action value")

        # terminate left
        if current_state == self.left_terminal_state:
            reward = -1.0
            is_terminal = True

        # terminate right
        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True

        else:
            reward = 0.0
            is_terminal = False

        return reward, current_state, is_terminal
