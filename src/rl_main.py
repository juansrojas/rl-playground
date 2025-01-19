import pandas as pd
import numpy as np
import copy


class ReinforcementLearning:
    def __init__(self, agent_class, environment_class, state_representation=None, track_data=True):
        # initialize instance variables
        self.agent = None
        self.environment = None

        # keep track of episode and step counts
        self.step_num = 0
        self.episode_num = 0

        # initialize agent
        self.agent = copy.deepcopy(agent_class)

        # initialize environment
        self.environment = copy.deepcopy(environment_class)

        # initialize state representation function
        self.state_representation = copy.deepcopy(state_representation)

        # keep track of RL data
        self.track_data = track_data
        self.data = []

    def get_data(self):
        # return RL data
        return pd.DataFrame(self.data)

    def rl_start(self, init_state=None, init_action=None):
        # return initial state and action to start the episode

        # get initial state
        state = self.environment.env_start(init_state)

        # get initial action
        action = self.agent.agent_start(init_state, init_action)

        # update episode and step counts
        self.episode_num += 1
        self.step_num = 0

        # return initial state and action
        return state, action

    def rl_step(self, last_state, last_action, epsilon=None, step_size=None, discount=None,
                planning=False, planning_steps=0, experience_replay=False, experience_replay_steps=0):

        # get reward and new state from environment based on last state and last action taken
        (reward, state, terminal) = self.environment.env_step(last_state, last_action)

        # get state
        if not pd.isnull(self.state_representation):
            # use function approximation if needed
            use_last_state = self.state_representation(last_state)
            use_state = self.state_representation(state)

        elif isinstance(state, tuple):
            # if tuple convert to numpy array
            use_last_state = np.array([last_state]).T
            use_state = np.array([state]).T

        else:
            use_last_state = last_state
            use_state = state

        # perform update
        self.step_num += 1
        if planning:
            action = self.agent.agent_step_with_planning(self.environment, use_last_state, last_action, use_state, reward, terminal, epsilon, step_size, discount, planning_steps)

        elif experience_replay:
            action = self.agent.agent_step_with_experience_replay(use_last_state, last_action, use_state, reward, terminal, epsilon, step_size, discount, experience_replay_steps)

        else:
            action = self.agent.agent_step(use_last_state, last_action, use_state, reward, terminal, epsilon, step_size, discount)

        if terminal:
            # if we reach a terminal state end episode
            action = None

        # track data
        if self.track_data:
            self.data.append({
                'episode': self.episode_num,
                'step': self.step_num,
                'state': last_state,
                'action': last_action,
                'reward': reward,
                'next_state': state,
                'next_action': action,
                'terminal': terminal,
            })

        return reward, state, action, terminal
