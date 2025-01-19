import pandas as pd


class EnvironmentCliffWalk:
    def __init__(self, grid_height, grid_width, default_reward=-1):
        # initialize instance variables
        self.grid_h = grid_height
        self.grid_w = grid_width

        # map states to locations
        self.state_dict = {}

        # starting location of agent is the bottom-left corner, (min x, min y).
        self.start_loc = (0, 0)

        # goal location is the bottom-right corner. (max x, min y).
        self.goal_loc = (self.grid_w - 1, 0)

        # the cliff will contain all the cells between the start_loc and goal_loc.
        self.cliff = [(i, 0) for i in range(1, (self.grid_w - 1))]

        # set default reward
        self.default_reward = default_reward

    def get_state(self, xy_location):
        # state = h * grid_width + w
        state = xy_location[1] * self.grid_w + xy_location[0]
        if not self.state_dict.get(state):
            self.state_dict[state] = xy_location
        return state

    @staticmethod
    def within_bounds(x, y, width, height):
        # check if within bounds
        if 0 <= x <= width - 1 and 0 <= y <= height - 1:
            return True
        else:
            return False

    def env_start(self, start_location=None):
        # return initial state
        if pd.isnull(start_location):
            return self.get_state(self.start_loc)
        else:
            return self.get_state(start_location)

    def env_step(self, state, action):
        # return new state and reward

        # get agent's current location
        x, y = self.state_dict[state]

        # determine the agent's new location based on the action taken and current location
        if action == 'up':
            y = y + 1

        elif action == 'left':
            x = x - 1

        elif action == 'down':
            y = y - 1

        elif action == 'right':
            x = x + 1

        else:
            raise Exception(str(action) + " not in recognized actions!")

        # if the action takes the agent out-of-bounds then the agent stays in the same state
        if not self.within_bounds(x, y, self.grid_w, self.grid_h):
            x, y = self.state_dict[state]

        # if the agent falls off the cliff and agent location is reset
        if (x, y) in self.cliff:
            reward = -100
            terminal = False
            (x, y) = self.start_loc

        # if the agent reaches the goal state
        elif (x, y) == self.goal_loc:
            reward = self.default_reward
            terminal = True

        # otherwise assume default reward and that we did not terminate
        else:
            reward = self.default_reward
            terminal = False

        return reward, self.get_state((x, y)), terminal
