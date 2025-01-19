import numpy as np


class EnvironmentBandit:
    def __init__(self, n_arms):
        # generate n arms that have a randomly-generated reward
        self.arms = np.random.randn(n_arms)

    @staticmethod
    def env_start(start_state):
        # return initial observation
        return start_state

    def env_step(self, state, action):
        # get the reward for a given arm with some noise
        reward = self.arms[action-1] + np.random.randn()

        # set new state and termination bool
        new_state = state
        termination = False

        return reward, new_state, termination
