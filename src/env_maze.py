import pandas as pd
import numpy as np
import plotly.graph_objects as go


class EnvironmentMaze:
    def __init__(self, grid_height, grid_width, obstacle_switch_time=np.inf, default_reward=-1):
        # initialize instance variables
        self.grid_h = grid_height
        self.grid_w = grid_width

        # map states to locations
        self.state_dict = {}

        # start state
        self.start_loc = (3, 0)

        # goal state
        self.goal_loc = (self.grid_w - 1, self.grid_h - 1)

        # obstacles
        self.obstacles = []

        # time to change obstacles
        self.obstacle_switch_time = obstacle_switch_time

        # timer so we know when to change obstacles
        self.counter = 0

        # so we know when to use generate shortcut
        self.shortcut = False

        # default reward
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

    def set_obstacles(self, random=True, shortcut=False):
        self.obstacles = []

        # generate random obstacle
        if random:
            if np.random.random() < 0.5:
                axis = 'height'
            else:
                axis = 'width'

            # obstacle will cover all but one location along axis
            if axis == 'height':
                not_blocked = np.random.choice(list(range(self.grid_h)))

                x = np.random.choice(list(range(self.grid_w)))
                while x in [self.start_loc[0], self.goal_loc[0]]:
                    x = np.random.choice(list(range(self.grid_w)))

                for i in range(self.grid_h):
                    if i != not_blocked and (x, i) != self.start_loc and (x, i) != self.goal_loc:
                        self.obstacles.append(self.get_state((x, i)))

            elif axis == 'width':
                not_blocked = np.random.choice(list(range(self.grid_w)))

                y = np.random.choice(list(range(self.grid_h)))
                while y in [self.start_loc[1], self.goal_loc[1]]:
                    y = np.random.choice(list(range(self.grid_h)))

                for i in range(self.grid_w):
                    if i != not_blocked and (i, y) != self.start_loc and (i, y) != self.goal_loc:
                        self.obstacles.append(self.get_state((i, y)))

        # generate pre-defined obstacle
        else:
            y = 2
            not_blocked = 0
            for i in range(self.grid_w):
                if i != not_blocked:
                    if not shortcut:
                        self.obstacles.append(self.get_state((i, y)))
                    else:
                        if (i, y) != (self.grid_w - 1, y):
                            self.obstacles.append(self.get_state((i, y)))

    def reset_counter(self):
        # reset counter at zero
        self.counter = 0

    def env_start(self, start_location=None):
        # initialize obstacles
        self.set_obstacles(random=False, shortcut=self.shortcut)

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

        # check if the agent hit an obstacle
        if self.get_state((x, y)) in self.obstacles:
            x, y = self.state_dict[state]

        # if the action takes the agent out-of-bounds then the agent stays in the same state
        if not self.within_bounds(x, y, self.grid_w, self.grid_h):
            x, y = self.state_dict[state]

        # assign reward
        if (x, y) == self.goal_loc:
            terminal = True
            reward = 0.0
        else:
            terminal = False
            reward = self.default_reward

        self.counter += 1
        if self.counter == self.obstacle_switch_time:
            self.shortcut = True

        return reward, self.get_state((x, y)), terminal

    def plot_env(self):
        print('Start state: ' + str(self.start_loc))
        print('Goal state: ' + str(self.goal_loc))

        x = []
        y = []
        z = []
        for h in range(self.grid_h):
            x.append(h)
            z.append([])
            for w in range(self.grid_w):
                y.append(w)

                if self.get_state((w, h)) in self.obstacles:
                    z[-1].append(1)
                else:
                    z[-1].append(0)

        fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale='gray', reversescale=True, showscale=False, xgap=2, ygap=2))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()
        return
