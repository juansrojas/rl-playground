import numpy as np

import src.tiles as tc


class StateRepresentation:
    def __init__(self, num_states=None, num_states_in_group=None, num_groups=None, num_tiles=None, num_tilings=None,
                 iht_size=None, min_pose=None, max_pose=None, min_vel=None, max_vel=None):
        # initialize instance variables

        # one hot encoding
        self.num_states = num_states

        # state aggregation
        self.num_states_in_group = num_states_in_group
        self.num_groups = num_groups

        # tile coding
        self.iht = tc.IHT(iht_size)
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings

        # bounds
        self.min_pose = min_pose
        self.max_pose = max_pose
        self.min_vel = min_vel
        self.max_vel = max_vel

    def one_hot_encode(self, state):
        # Create the one-hot encoding of state (1, num_states)
        one_hot_vector = self.num_states * [0]
        one_hot_vector[int((state - 1))] = 1
        return np.array([one_hot_vector]).T

    def state_aggregation(self, state):
        # Generate one-hot encoded state feature vector

        # Example:
        # If num_states = 100, num_states_in_group = 20, num_groups = 5,
        # one_hot_vector would be of size 5.
        # For states 1~20, one_hot_vector would be: [1, 0, 0, 0, 0]

        feature_vector = self.num_groups * [0]
        if state % self.num_states_in_group == 0:
            feature_vector[int(state / self.num_states_in_group - 1)] = 1
        else:
            feature_vector[int(np.floor(state / self.num_states_in_group))] = 1

        return np.array([feature_vector]).T

    def dynamicstate_tilecoding(self, state):
        # Takes in a position and velocity and returns a numpy array of active tiles.

        position, velocity = state

        # scale position and velocity to the range [0, 1]
        # then multiply that range with self.num_tiles, so it scales from [0, num_tiles]
        position_scaled = ((position - self.min_pose) / (self.max_pose - self.min_pose)) * self.num_tiles
        velocity_scaled = ((velocity - self.min_vel) / (self.max_vel - self.min_vel)) * self.num_tiles

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        tiles = tc.tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])

        feature_vector = np.zeros((self.iht.size, 1))
        feature_vector[tiles] = 1

        return feature_vector
