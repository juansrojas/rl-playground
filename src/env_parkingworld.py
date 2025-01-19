import numpy as np


class ParkingWorld:
    def __init__(self, num_spaces, num_prices, price_factor=0.1, occupants_factor=1.0, null_factor=1/3):
        # initialize instance variables
        self.num_spaces = num_spaces
        self.num_prices = num_prices

        self.occupants_factor = occupants_factor
        self.price_factor = price_factor
        self.null_factor = null_factor

        # initialize states and actions
        self.states = [num_occupied for num_occupied in range(num_spaces + 1)]
        self.actions = list(range(num_prices))

    def state_reward(self, state):
        # get reward for being in a given state
        if state == self.num_spaces:
            return self.null_factor * state * self.occupants_factor
        else:
            return state * self.occupants_factor

    def get_reward(self, last_state, last_action, state):
        # get the reward from transitioning to a given state
        return self.state_reward(last_state) + self.state_reward(state)

    def get_state_probability(self, state, last_state, last_action):
        # get P(S' | S, A)
        center = (1 - self.price_factor) * last_state + self.price_factor * self.num_spaces * (1 - last_action / self.num_prices)
        emphasis = np.exp(-abs(np.arange(2 * self.num_spaces) - center) / 5)
        if state == self.num_spaces:
            return sum(emphasis[state:]) / sum(emphasis)
        return emphasis[state] / sum(emphasis)

    def get_transitions(self, last_state, last_action):
        # get possible transition rewards and probabilities
        transitions = {}
        for state in self.states:
            transitions[state] = {}
            transitions[state]['reward'] = self.get_reward(last_state, last_action, state)
            transitions[state]['probability'] = self.get_state_probability(state, last_state, last_action)
        return transitions

    def env_step(self, last_state, last_action):
        probabilities = []
        for state in self.states:
            probabilities.append(self.get_state_probability(state, last_state, last_action))
        new_state = np.random.choice(self.states, p=probabilities)
        reward = self.get_reward(last_state, last_action, new_state)
        terminal = False
        return reward, new_state, terminal
