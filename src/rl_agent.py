import pandas as pd
import numpy as np
import copy

import tensorflow as tf
import torch


class RLAgent:
    def __init__(self, agent_type, states, actions, policy=None, use_average_reward=False,
                 use_value_trace=False, value_trace_lambda=0, use_policy_trace=False, policy_trace_lambda=0,
                 policy_type='tabular', policy_softmax_tau=1.0, policy_nn_config=None, policy_update_type='stochastic_gradient_descent',
                 value_type='tabular', value_nn_config=None, value_update_type='stochastic_gradient_descent',
                 beta_m_adam=0.9, beta_v_adam=0.999, epsilon_adam=1e-8,
                 planning_type=None, planning_kappa=None, planning_model=None,
                 use_experience_replay=False, replay_buffer_size=None, replay_buffer_minibatch_size=None):

        # initialize instance variables
        self.agent_type = agent_type

        self.states = states
        self.actions = actions

        # average reward
        self.use_avg_reward = use_average_reward
        if self.use_avg_reward:
            self.avg_reward = 0.0

        # eligibility traces
        self.use_value_trace = use_value_trace
        self.value_trace_lambda = value_trace_lambda
        self.use_policy_trace = use_policy_trace
        self.policy_trace_lambda = policy_trace_lambda
        self.value_trace = {}
        self.policy_trace = {}

        # raise eligibility trace errors
        if use_value_trace and agent_type == 'expected_sarsa':
            raise Exception('ERROR: eligibility traces are not defined for expected sarsa')

        if use_policy_trace and policy_type == 'tabular':
            raise Exception('ERROR: eligibility traces are not defined for tabular policies')

        if use_experience_replay and (use_value_trace or use_policy_trace):
            raise Exception('ERROR: Experience replay and eligibility traces are both enabled')

        if (use_value_trace or use_policy_trace) and (value_type in ['nn_pytorch', 'nn_keras'] or policy_type in ['nn_pytorch', 'nn_keras']):
            raise Exception('ERROR: eligibility traces are not implemented with keras or pytorch')

        # bandits
        if agent_type == 'bandit':
            self.arm_count = {}
            for action in self.actions:
                self.arm_count[action] = 0

        # planning
        self.planning_type = planning_type
        if not pd.isnull(planning_type):
            if self.planning_type == 'dyna_plus':
                self.planning_kappa = planning_kappa

                self.planning_tau = {}
                for state in self.states:
                    self.planning_tau[state] = {}
                    for action in self.actions:
                        self.planning_tau[state][action] = 0

            if pd.isnull(planning_model):
                self.planning_model = {}
            else:
                self.planning_model = planning_model

        # experience replay
        self.use_experience_replay = use_experience_replay
        if self.use_experience_replay:
            self.replay_buffer = []
            self.replay_buffer_size = replay_buffer_size
            self.replay_buffer_minibatch_size = replay_buffer_minibatch_size

        #################################
        # state and state-action values
        #################################
        self.value_type = value_type
        self.value_update_type = value_update_type

        # tabular case
        if self.value_type == 'tabular':
            self.values = {}
            if self.agent_type in ['bandit']:
                for action in self.actions:
                    self.values[action] = 0

            elif self.agent_type in ['dp', 'td']:
                for state in self.states:
                    self.values[state] = 0

            elif self.agent_type in ['q_learning', 'sarsa', 'expected_sarsa']:
                for state in self.states:
                    self.values[state] = {}
                    for action in self.actions:
                        self.values[state][action] = 0

        # function approximation
        else:
            self.value_nn_config = value_nn_config
            self.beta_m = beta_m_adam
            self.beta_v = beta_v_adam
            self.beta_m_product = beta_m_adam
            self.beta_v_product = beta_v_adam
            self.epsilon_adam = epsilon_adam

            self.neural_network_value = None
            self.weights_value = {}
            self.m_value = {}
            self.v_value = {}

            # initialize weights
            self.neural_network_value, self.weights_value, self.m_value, self.v_value = self.initialize_weights(weights_type='value_approx', nn_config=self.value_nn_config)

            if self.use_experience_replay:
                # generate target network for experience replay
                if self.value_type == 'neural_network':
                    self.weights_target_value = copy.deepcopy(self.weights_value)
                elif self.value_type in ['nn_pytorch', 'nn_keras']:
                    self.neural_network_target_value = copy.deepcopy(self.neural_network_value)

        ################
        # agent policy
        ################
        self.policy_type = policy_type
        self.policy_softmax_tau = policy_softmax_tau
        self.policy_update_type = policy_update_type
        self.policy_discount = None

        # tabular
        if self.policy_type == 'tabular':
            self.policy = policy

        # softmax (non-parameterized)
        elif self.policy_type == 'softmax':
            # no need to do anything :)
            pass

        # softmax parameterized
        else:
            if pd.isnull(value_type):
                # initialize adam update params if we haven't done so yet
                self.beta_m = beta_m_adam
                self.beta_v = beta_v_adam
                self.beta_m_product = beta_m_adam
                self.beta_v_product = beta_v_adam
                self.epsilon_adam = epsilon_adam

            self.policy_nn_config = policy_nn_config

            self.neural_network_policy = None
            self.weights_policy = {}
            self.m_policy = {}
            self.v_policy = {}

            # initialize weights
            self.neural_network_policy, self.weights_policy, self.m_policy, self.v_policy = self.initialize_weights(weights_type='policy', nn_config=self.policy_nn_config)

    def initialize_weights(self, weights_type, nn_config=None, initial_nn_weights_value=None):
        # initialize weights
        if weights_type == 'value_approx' and self.agent_type == 'td':
            value_dimension = 1
        else:
            value_dimension = len(self.actions)

        if weights_type == 'value_approx':
            model_type = self.value_type
        elif weights_type == 'policy':
            model_type = self.policy_type

        neural_network = None
        weights = {}
        m = {}
        v = {}
        if model_type == 'linear':
            weights[weights_type] = {
                'W': np.zeros((value_dimension, len(self.states))),
            }

            m[weights_type] = {
                'W': np.zeros((value_dimension, len(self.states))),
            }

            v[weights_type] = {
                'W': np.zeros((value_dimension, len(self.states))),
            }

        elif model_type == 'neural_network':
            # Note: for estimating action-values, output layer should have the same number of nodes as actions
            counter = 0
            for layer in nn_config['layers'].keys():
                # input layer does not need weights
                if counter >= 1:
                    if pd.isnull(initial_nn_weights_value):
                        # if no initial weights value is given, initialize weights randomly
                        weights[layer] = {
                            'W': np.random.normal(0, np.sqrt(2 / nn_config['layers'][prev_layer]['nodes']), size=(nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes'])),
                            'b': np.random.normal(0, np.sqrt(2 / nn_config['layers'][prev_layer]['nodes']), size=(nn_config['layers'][layer]['nodes'], 1)),
                        }
                    else:
                        weights[layer] = {
                            'W': initial_nn_weights_value * np.ones((nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes'])),
                            'b': initial_nn_weights_value * np.ones((nn_config['layers'][layer]['nodes'], 1)),
                        }

                    m[layer] = {
                        'W': np.zeros((nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes'])),
                        'b': np.zeros((nn_config['layers'][layer]['nodes'], 1)),
                    }

                    v[layer] = {
                        'W': np.zeros((nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes'])),
                        'b': np.zeros((nn_config['layers'][layer]['nodes'], 1)),
                    }

                prev_layer = layer
                counter += 1

        elif model_type == 'nn_pytorch':
            neural_network = {}

            # create network
            layers = []
            counter = 0
            for layer in nn_config['layers'].keys():
                if counter >= 1:
                    if nn_config['layers'][layer]['activation_function'] == 'linear':
                        layers.append(torch.nn.Linear(nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes']))
                    elif nn_config['layers'][layer]['activation_function'] == 'relu':
                        layers.append(torch.nn.Linear(nn_config['layers'][prev_layer]['nodes'], nn_config['layers'][layer]['nodes']))
                        layers.append(torch.nn.ReLU(inplace=False))

                prev_layer = layer
                counter += 1

            if weights_type == 'policy':
                layers.append(torch.nn.Softmax(dim=1))

            neural_network['network'] = torch.nn.Sequential(*layers)

            # create mean squared error value loss function
            if weights_type == 'value_approx':
                neural_network['value_loss_function'] = torch.nn.MSELoss()

            # create optimizer
            if weights_type == 'value_approx':
                optimizer_type = self.value_update_type
            elif weights_type == 'policy':
                optimizer_type = self.policy_update_type

            # use placeholder learning rate for now (it will be set as the user-defined step size during learning)
            if optimizer_type == 'stochastic_gradient_descent':
                neural_network['optimizer'] = torch.optim.SGD(neural_network['network'].parameters(), lr=1e-3)

            elif optimizer_type == 'adam':
                neural_network['optimizer'] = torch.optim.Adam(neural_network['network'].parameters(), lr=1e-3,
                                                               betas=(self.beta_m, self.beta_v),
                                                               eps=self.epsilon_adam)

        elif model_type == 'nn_keras':
            neural_network = {}

            # create network
            neural_network['network'] = tf.keras.models.Sequential()
            counter = 0
            for layer in nn_config['layers'].keys():
                if counter >= 1:
                    neural_network['network'].add(tf.keras.layers.Dense(nn_config['layers'][layer]['nodes'],
                                                                        input_dim=nn_config['layers'][prev_layer]['nodes'],
                                                                        activation=nn_config['layers'][layer]['activation_function']))

                prev_layer = layer
                counter += 1

            if weights_type == 'policy':
                neural_network['network'].add(tf.keras.layers.Softmax(input_dim=nn_config['layers'][layer]['nodes']))

            # create mean squared error value loss function
            if weights_type == 'value_approx':
                neural_network['value_loss_function'] = tf.keras.losses.MeanSquaredError()

            # create optimizer
            if weights_type == 'value_approx':
                optimizer_type = self.value_update_type
            elif weights_type == 'policy':
                optimizer_type = self.policy_update_type

            # use placeholder learning rate for now (it will be set as the user-defined step size during learning)
            if optimizer_type == 'stochastic_gradient_descent':
                neural_network['optimizer'] = tf.keras.optimizers.legacy.SGD(learning_rate=1e-3)
            elif optimizer_type == 'adam':
                neural_network['optimizer'] = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3,
                                                                              beta_1=self.beta_m,
                                                                              beta_2=self.beta_v,
                                                                              epsilon=self.epsilon_adam)

        return neural_network, weights, m, v

    def reset_eligibility_traces(self, reset_value_trace=True, reset_policy_trace=True):
        # reset eligibility traces
        if self.use_value_trace and reset_value_trace:
            if self.value_type == 'tabular':
                # tabular case
                if self.agent_type == 'td':
                    for state in self.states:
                        self.value_trace[state] = 0
                else:
                    for state in self.states:
                        self.value_trace[state] = {}
                        for action in self.actions:
                            self.value_trace[state][action] = 0
            else:
                # value function approximation
                # eligibility trace will have the same form as weights
                _, self.value_trace, _, _ = self.initialize_weights(weights_type='value_approx', nn_config=self.value_nn_config, initial_nn_weights_value=0)

        if self.use_policy_trace and reset_policy_trace:
            # parameterized policy
            # eligibility trace will have the same form as weights
            _, self.policy_trace, _, _ = self.initialize_weights(weights_type='policy', nn_config=self.policy_nn_config, initial_nn_weights_value=0)

        return

    @staticmethod
    def argmax(values):
        # get argmax of values, breaking ties arbitrarily

        top_value = float("-inf")
        ties = []
        for action in values.keys():
            q = values[action]
            if q > top_value:
                top_value = q
                ties = [action]
            elif q == top_value:
                ties.append(action)
        return np.random.choice(ties)

    def bellman_update(self, env, state, discount):
        # update values according to the Bellman update equation

        v_new = 0
        for a in range(len(self.actions)):
            pi_conditional = self.policy[state][a]
            transitions = env.get_transitions(state, a)
            for s_prime in range(len(self.states)):
                r = transitions[s_prime]['reward']
                p = transitions[s_prime]['probability']
                v_new += pi_conditional * p * (r + discount * self.values[s_prime])

        self.values[state] = v_new

        return

    def bellman_optimality_update(self, env, state, discount):
        # update values according to the Bellman optimality update equation

        v_max = 0
        for a in range(len(self.actions)):
            v = 0
            transitions = env.get_transitions(state, a)
            for s_prime in range(len(self.states)):
                r = transitions[s_prime]['reward']
                p = transitions[s_prime]['probability']
                v += p * (r + discount * self.values[s_prime])

            if v > v_max:
                v_max = v

        self.values[state] = v_max

        return

    def greedify_policy(self, env, state, discount):
        # update policy to be greedy with respect to the state-action values

        q_max = 0
        action_max = None
        for a in range(len(self.actions)):
            q = 0
            transitions = env.get_transitions(state, a)
            for s_prime in range(len(self.states)):
                r = transitions[s_prime]['reward']
                p = transitions[s_prime]['probability']
                q += p * (r + discount * self.values[s_prime])

            if q > q_max:
                q_max = q
                action_max = a

        for a in range(len(self.actions)):
            if a == action_max:
                self.policy[state][a] = 1
            else:
                self.policy[state][a] = 0

        return

    def dp_policy_evaluation(self, env, discount, theta):
        value_delta = float('inf')
        while value_delta > theta:
            value_delta = 0
            for state in self.states:
                v = self.values[state]
                self.bellman_update(env, state, discount)
                value_delta = max(value_delta, abs(v - self.values[state]))

        return

    def dp_policy_iteration(self, env, discount, theta):
        policy_stable = False
        while not policy_stable:
            self.dp_policy_evaluation(env, discount, theta)

            # improve policy
            policy_stable = True
            for state in self.states:
                old = self.policy[state].copy()
                self.greedify_policy(env, state, discount)

                # check if policies are the same
                if self.policy[state] != old:
                    policy_stable = False

        return

    def dp_value_iteration(self, env, discount, theta):
        delta = float('inf')
        while delta > theta:
            delta = 0
            for state in self.states:
                v = self.values[state]
                self.bellman_optimality_update(env, state, discount)
                delta = max(delta, abs(v - self.values[state]))

        for state in self.states:
            self.greedify_policy(env, state, discount)

        return

    def nn_activation(self, x_vector, layer, nn_type):
        # activation functions for neural network

        if nn_type == 'value_approx':
            nn_config = self.value_nn_config
        elif nn_type == 'policy':
            nn_config = self.policy_nn_config

        if nn_config['layers'][layer]['activation_function'] == 'linear':
            return x_vector

        elif nn_config['layers'][layer]['activation_function'] == 'relu':
            return np.maximum(x_vector, 0)

    def nn_forward_pass(self, weights, state, nn_type):
        x_vectors = {'input': state}
        x_activated = x_vectors['input']
        for layer in weights.keys():
            x = np.matmul(weights[layer]['W'].T, x_activated) + weights[layer]['b']
            x_activated = self.nn_activation(x, layer, nn_type)
            x_vectors[layer] = x_activated
        return x_vectors

    def get_value(self, state, action=None):
        # returns the state or state-action value

        # tabular case
        if self.value_type == 'tabular':
            if self.agent_type in ['bandit']:
                return self.values[action]
            elif self.agent_type in ['dp', 'td']:
                return self.values[state]
            elif self.agent_type in ['q_learning', 'sarsa', 'expected_sarsa']:
                return self.values[state][action]

        # value-function approximation
        else:
            if not pd.isnull(action):
                value_index = self.actions.index(action)
            else:
                value_index = 0

            if self.value_type == 'linear':
                return np.dot(self.weights_value['value_approx']['W'][value_index], state)[0]

            elif self.value_type == 'neural_network':
                return self.nn_forward_pass(self.weights_value, state, nn_type='value_approx')['output'][value_index][0]

            elif self.value_type == 'nn_pytorch':
                return self.neural_network_value['network'](torch.from_numpy(state.T).float())[0][value_index].item()

            elif self.value_type == 'nn_keras':
                return self.neural_network_value['network'].predict(state.T, verbose=0)[0][value_index]

    def get_target_value(self, state, action=None):
        # returns the target state or state-action value

        if not self.use_experience_replay:
            # if we are not using experience replay, return regular value
            return self.get_value(state, action)
        else:
            # if we are using experience replay, use target network to generate target value
            if not pd.isnull(action):
                value_index = self.actions.index(action)
            else:
                value_index = 0

            if self.value_type == 'neural_network':
                return self.nn_forward_pass(self.weights_target_value, state, nn_type='value_approx')['output'][value_index][0]

            if self.value_type == 'nn_pytorch':
                return self.neural_network_target_value['network'](torch.from_numpy(state.T).float())[0][value_index].item()

            elif self.value_type == 'nn_keras':
                return self.neural_network_target_value['network'].predict(state.T, verbose=0)[0][value_index]

    def get_softmax_probabilities(self, state):
        # computes the state-action preferences

        if self.policy_type == 'nn_pytorch':
            softmax_probs = []
            for a in range(len(self.actions)):
                softmax_probs.append(self.neural_network_policy['network'](torch.from_numpy(state.T).float())[0][a].item())

            # normalize softmax probabilities to make sure that they add up to exactly 1.0
            softmax_probs = np.array(softmax_probs, dtype=np.float64)
            softmax_probs = softmax_probs / softmax_probs.sum()

        elif self.policy_type == 'nn_keras':
            softmax_probs = []
            for a in range(len(self.actions)):
                softmax_probs.append(self.neural_network_policy['network'].predict(state.T, verbose=0)[0][a])

            # normalize softmax probabilities to make sure that they add up to exactly 1.0
            softmax_probs = np.array(softmax_probs, dtype=np.float64)
            softmax_probs = softmax_probs / softmax_probs.sum()

        else:
            state_action_preferences = []
            for a in range(len(self.actions)):
                if self.policy_type == 'softmax':
                    # if no parameterized policy, use action-state values as action preferences
                    state_action_preferences.append(self.get_value(state, self.actions[a]) / self.policy_softmax_tau)

                elif self.policy_type == 'linear':
                    state_action_preferences.append(np.dot(self.weights_policy['policy']['W'][a], state)[0])

                elif self.policy_type == 'neural_network':
                    state_action_preferences.append(self.nn_forward_pass(self.weights_policy, state, nn_type='policy')['output'][a][0])

            # Set the constant c by finding the maximum of state-action preferences
            c = np.max(state_action_preferences)

            # get numerator by subtracting c from state-action preferences and exponentiating it
            numerator = []
            for h in state_action_preferences:
                numerator.append(np.exp(h - c))

            # get denominator by summing the values in the numerator
            denominator = np.sum(numerator)

            # get action probs by dividing each element in numerator array by denominator
            softmax_probs = []
            for n in numerator:
                softmax_probs.append(n / denominator)

        # get policy dictionary
        i = 0
        softmax_dict = {}
        for a in self.actions:
            softmax_dict[a] = softmax_probs[i]
            i += 1

        return softmax_dict

    def get_policy(self, state, action, argmax_action=None, epsilon=None, softmax_probs=None):
        # returns the probability of taking action A at state S

        if self.policy_type == 'tabular':
            if pd.isnull(self.policy):
                # epsilon greedy policy
                if action == argmax_action:
                    return (1 - epsilon) + (epsilon / len(self.actions))
                else:
                    return epsilon / len(self.actions)
            else:
                # user-defined tabular policy
                return self.policy[state][action]
        else:
            # softmax, possibly parameterized, policy
            return softmax_probs[action]

    def choose_action_from_policy(self, state, epsilon):
        # chooses action based on policy

        if self.policy_type == 'tabular':
            if not pd.isnull(self.policy):
                # choose action using user-defined tabular policy
                action_probs = []
                for a in self.actions:
                    action_probs.append(self.policy[state][a])
                action = np.random.choice(self.actions, p=action_probs)
            else:
                # choose action using epsilon greedy policy
                if np.random.random() <= epsilon:
                    action = np.random.choice(self.actions)
                else:
                    # tabular case
                    if self.value_type == 'tabular':
                        if self.agent_type == 'bandit':
                            values = self.values
                        else:
                            values = self.values[state]

                    # function approximation
                    else:
                        values = {}
                        for action in self.actions:
                            values[action] = self.get_value(state, action)

                    action = self.argmax(values)
        else:
            # choose action using softmax, possibly parameterized, policy
            action = np.random.choice(self.actions, p=list(self.get_softmax_probabilities(state).values()))

        return action

    def get_gradient(self, state, weights, gradient_type, weight_type, nn_config=None, action=None):
        # get the value-function approximation gradient

        if weight_type == 'value_approx' and self.agent_type == 'td':
            value_index = 0
        else:
            value_index = self.actions.index(action)

        if gradient_type == 'linear':
            gradient = np.zeros(weights[weight_type]['W'].shape)
            gradient[value_index] = state.T
            gradients = {
                weight_type: {
                    'W': gradient,
                }
            }
            return gradients

        elif gradient_type == 'neural_network':
            # calculate gradient for neural network layers with respect to the value:
            gradients = {}

            # get intermediate values
            x_vectors = self.nn_forward_pass(weights, state, nn_type=weight_type)

            # backwards propagation
            layers = list(reversed(x_vectors.keys()))
            for layer_index in range(len(layers) - 1):
                prev_layer = layers[layer_index + 1]
                layer = layers[layer_index]

                x_prev = x_vectors[prev_layer]
                x = x_vectors[layer]

                if layer_index == 0:
                    # output delta param is 1 since we are taking the gradient with respect to the value
                    # for state-action values, only the selected action has a value of 1
                    delta = np.zeros((1, len(x)))
                    delta[0][value_index] = 1

                    # there are no weights after the output
                    x_weights = np.identity(len(x))

                # gradient for linear activation function
                if nn_config['layers'][layer]['activation_function'] == 'linear':
                    x_activated_gradient = np.identity(len(x))

                # gradient for ReLU activation function
                elif nn_config['layers'][layer]['activation_function'] == 'relu':
                    x_activated_gradient = np.identity(len(x))
                    for i in range(len(x)):
                        if x[i] <= 0:
                            x_activated_gradient[i, i] = 0

                delta_prev = np.matmul(np.matmul(delta, x_weights.T), x_activated_gradient)

                gradients[layer] = {
                    'W': np.matmul(x_prev, delta_prev),
                    'b': delta_prev.T,
                }

                delta = delta_prev
                x_weights = weights[layer]['W']

            return gradients

    def get_softmax_policy_gradient(self, state, action, weights, gradient_type, weight_type, nn_config=None):
        # calculate gradient of softmax, parameterized policy

        # get softmax probabilities
        softmax_probs = self.get_softmax_probabilities(state)

        # get gradient of action preferences parameterization
        all_gradients = {}
        for a in self.actions:
            all_gradients[a] = self.get_gradient(state=state,
                                                 action=a,
                                                 weights=weights,
                                                 gradient_type=gradient_type,
                                                 weight_type=weight_type,
                                                 nn_config=nn_config)

        # get softmax gradient
        # grad_ln(pi(a | s, theta)) = grad_preferences(s, a, theta) - sum_i[pi(i)*grad_preferences(s, i, theta)]
        gradients = {}
        for item in all_gradients[action].keys():
            gradients[item] = {}
            for param in all_gradients[action][item].keys():
                gradients[item][param] = np.array(all_gradients[action][item][param])

                # subtract terms
                for action_index in range(len(self.actions)):
                    gradients[item][param][action_index] -= softmax_probs[self.actions[action_index]] * all_gradients[self.actions[action_index]][item][param][action_index]

        return gradients

    def update_weights(self, weights_type, weights, step_size, delta, gradient, eligibility_trace, m, v, update_type,
                       use_trace=False, trace_lambda=None, policy_discount=None):

        # update weights and eligibility traces
        if weights_type == 'policy':
            gradient = policy_discount * gradient

        if use_trace:
            eligibility_trace = trace_lambda * eligibility_trace + gradient
            use_gradient = eligibility_trace
        else:
            eligibility_trace = None
            use_gradient = gradient

        if update_type == 'stochastic_gradient_descent':
            weights = weights + step_size * delta * use_gradient
            m = m
            v = v

        elif update_type == 'adam':
            # compute g
            g = delta * use_gradient

            # update m and v
            m = self.beta_m * m + (1 - self.beta_m) * g
            v = self.beta_v * v + (1 - self.beta_v) * (g * g)

            # compute m_hat and v_hat
            m_hat = m / (1 - self.beta_m_product)
            v_hat = v / (1 - self.beta_v_product)

            # update weights
            weights = weights + step_size * m_hat / (np.sqrt(v_hat) + self.epsilon_adam)

        return weights, m, v, eligibility_trace

    def value_tabular_update(self, last_state, last_action, action, step_size, discount, estimate, target, argmax_action):
        # update tabular values and eligibility traces
        delta = target - estimate
        if self.agent_type in ['bandit']:
            self.values[last_action] = self.values[last_action] + step_size['value'] * delta

        elif self.agent_type in ['dp', 'td']:
            if not self.use_value_trace:
                # regular update without eligibility trace
                self.values[last_state] = self.values[last_state] + step_size['value'] * delta
            else:
                # update eligibility trace
                self.value_trace[last_state] += 1

                # update value using eligibility trace
                for s in self.states:
                    if self.value_trace.get(s):
                        self.values[s] = self.values[s] + step_size['value'] * delta * self.value_trace[s]

                    # decay trace
                    if self.use_avg_reward:
                        self.value_trace[s] = self.value_trace_lambda * self.value_trace[s]
                    else:
                        self.value_trace[s] = discount * self.value_trace_lambda * self.value_trace[s]

        elif self.agent_type in ['q_learning', 'sarsa', 'expected_sarsa']:
            if not self.use_value_trace:
                # regular update without eligibility trace
                self.values[last_state][last_action] = self.values[last_state][last_action] + step_size['value'] * delta
            else:
                # update eligibility trace
                self.value_trace[last_state][last_action] += 1

                # update value using eligibility trace
                for s in self.states:
                    for a in self.actions:
                        if self.value_trace[s].get(a):
                            self.values[s][a] = self.values[s][a] + step_size['value'] * delta * self.value_trace[s][a]

                        # decay trace
                        if self.use_avg_reward:
                            self.value_trace[s][a] = self.value_trace_lambda * self.value_trace[s][a]
                        else:
                            self.value_trace[s][a] = discount * self.value_trace_lambda * self.value_trace[s][a]

                # Watkins's Q(lambda) for Q-learning
                if self.agent_type == 'q_learning' and action != argmax_action:
                    self.reset_eligibility_traces(reset_value_trace=True, reset_policy_trace=False)

        return

    def value_approx_update(self, last_state, last_action, action, step_size, estimate, target, argmax_action):
        # value function approximation update

        # pytorch learning step
        if self.value_type == 'nn_pytorch':
            # set step size for learning
            self.neural_network_value['optimizer'].param_groups[0]['lr'] = step_size['value']

            # get target
            if self.agent_type == 'td':
                value_index = 0
            else:
                value_index = self.actions.index(last_action)

            target_fit = self.neural_network_value['network'](torch.from_numpy(last_state.T).float()).detach().numpy()
            target_fit[0][value_index] = target
            target_fit = torch.from_numpy(target_fit)

            # clear the accumulated gradients
            self.neural_network_value['optimizer'].zero_grad()

            # get estimate
            estimate_fit = self.neural_network_value['network'](torch.from_numpy(last_state.T).float())

            # calculate loss on neural network output and target
            loss = self.neural_network_value['value_loss_function'](estimate_fit, target_fit)

            # backwards propagation to calculate the gradients
            loss.backward()

            # optimize the neural network parameters in the direction of the gradients
            self.neural_network_value['optimizer'].step()

            return

        # keras learning step
        elif self.value_type == 'nn_keras':
            # set step size for learning
            self.neural_network_value['optimizer'].learning_rate = step_size['value']

            # get target
            if self.agent_type == 'td':
                value_index = 0
            else:
                value_index = self.actions.index(last_action)

            target_fit = self.neural_network_value['network'].predict(last_state.T, verbose=0)
            target_fit[0][value_index] = target

            # open a GradientTape to enable auto-differentiation
            with tf.GradientTape() as tape:
                # run the forward pass of the layer to get estimate
                estimate_fit = self.neural_network_value['network'](last_state.T, training=True)

                # calculate loss on neural network output and target
                loss = self.neural_network_value['value_loss_function'](target_fit, estimate_fit)

            # use the gradient tape to automatically retrieve the gradients with respect to the loss
            gradients = tape.gradient(loss, self.neural_network_value['network'].trainable_weights)

            # optimize the neural network parameters in the direction of the gradients
            self.neural_network_value['optimizer'].apply_gradients(zip(gradients, self.neural_network_value['network'].trainable_weights))

            return

        else:
            # calculate gradient and then update weights and traces
            delta = target - estimate

            gradient = self.get_gradient(state=last_state,
                                         action=last_action,
                                         weights=self.weights_value,
                                         gradient_type=self.value_type,
                                         weight_type='value_approx',
                                         nn_config=self.value_nn_config)

            # update weights
            if self.value_type == 'linear':
                if self.agent_type == 'td':
                    value_index = slice(0, 1)
                else:
                    value_index = slice(self.actions.index(last_action), self.actions.index(last_action) + 1)
            elif self.value_type == 'neural_network':
                value_index = slice(None, None)

            for item in self.weights_value.keys():
                for param in self.weights_value[item].keys():
                    if self.use_value_trace:
                        eligibility_trace = self.value_trace[item][param][value_index]
                    else:
                        eligibility_trace = None

                    w, m, v, z = self.update_weights(weights_type='value',
                                                     weights=self.weights_value[item][param][value_index],
                                                     step_size=step_size['value'],
                                                     delta=delta,
                                                     gradient=gradient[item][param][value_index],
                                                     eligibility_trace=eligibility_trace,
                                                     m=self.m_value[item][param][value_index],
                                                     v=self.v_value[item][param][value_index],
                                                     update_type=self.value_update_type,
                                                     use_trace=self.use_value_trace,
                                                     trace_lambda=self.value_trace_lambda)

                    self.weights_value[item][param][value_index] = w
                    self.m_value[item][param][value_index] = m
                    self.v_value[item][param][value_index] = v

                    if self.use_value_trace:
                        self.value_trace[item][param][value_index] = z

            # Watkins's Q(lambda) for Q-learning
            if self.use_value_trace and self.agent_type == 'q_learning' and action != argmax_action:
                self.reset_eligibility_traces(reset_value_trace=True, reset_policy_trace=False)

            return

    def policy_update(self, last_state, last_action, step_size, discount, estimate, target):
        # parameterized policy update

        # pytorch learning step
        if self.policy_type == 'nn_pytorch':
            # set step size for learning
            self.neural_network_policy['optimizer'].param_groups[0]['lr'] = step_size['policy']

            # clear the accumulated gradients
            self.neural_network_policy['optimizer'].zero_grad()

            # calculate neural network loss
            softmax_probs = self.neural_network_policy['network'](torch.from_numpy(last_state.T).float())
            action_prob_dist = torch.distributions.Categorical(softmax_probs)
            loss = -action_prob_dist.log_prob(torch.tensor(self.actions.index(last_action))) * (target - estimate)

            # backwards propagation to calculate the gradients
            loss.backward()

            # optimize the neural network parameters in the direction of the gradients
            self.neural_network_policy['optimizer'].step()

            return

        # keras learning step
        elif self.policy_type == 'nn_keras':
            # set step size for learning
            self.neural_network_policy['optimizer'].learning_rate = step_size['policy']

            # open a GradientTape to enable auto-differentiation
            with tf.GradientTape() as tape:
                # calculate neural network loss
                softmax_probs = self.neural_network_policy['network'](last_state.T, training=True)
                loss = -tf.keras.backend.log(softmax_probs[0][self.actions.index(last_action)]) * (target - estimate)

            # use the gradient tape to automatically retrieve the gradients with respect to the loss
            gradients = tape.gradient(loss, self.neural_network_policy['network'].trainable_weights)

            # optimize the neural network parameters in the direction of the gradients
            self.neural_network_policy['optimizer'].apply_gradients(zip(gradients, self.neural_network_policy['network'].trainable_weights))

            return

        else:
            # calculate gradient and then update weights and traces
            delta = target - estimate
            policy_gradient = self.get_softmax_policy_gradient(state=last_state,
                                                               action=last_action,
                                                               weights=self.weights_policy,
                                                               gradient_type=self.policy_type,
                                                               weight_type='policy',
                                                               nn_config=self.policy_nn_config)

            # update policy weights
            if self.policy_type == 'linear':
                action_index = slice(self.actions.index(last_action), self.actions.index(last_action) + 1)
            elif self.policy_type == 'neural_network':
                action_index = slice(None, None)

            for item in self.weights_policy.keys():
                for param in self.weights_policy[item].keys():
                    if self.use_policy_trace:
                        eligibility_trace = self.policy_trace[item][param][action_index]
                    else:
                        eligibility_trace = None

                    w, m, v, z = self.update_weights(weights_type='policy',
                                                     weights=self.weights_policy[item][param][action_index],
                                                     step_size=step_size['policy'],
                                                     delta=delta,
                                                     gradient=policy_gradient[item][param][action_index],
                                                     policy_discount=self.policy_discount,
                                                     eligibility_trace=eligibility_trace,
                                                     m=self.m_policy[item][param][action_index],
                                                     v=self.v_policy[item][param][action_index],
                                                     update_type=self.policy_update_type,
                                                     trace_lambda=self.policy_trace_lambda,
                                                     use_trace=self.use_policy_trace)

                    self.weights_policy[item][param][action_index] = w
                    self.m_policy[item][param][action_index] = m
                    self.v_policy[item][param][action_index] = v

                    if self.use_policy_trace:
                        self.policy_trace[item][param][action_index] = z

            if not self.use_avg_reward:
                self.policy_discount = discount * self.policy_discount

            return

    def agent_start(self, init_state, init_action):
        # start episode

        # reset eligibility traces
        self.reset_eligibility_traces()

        # reset policy discount
        self.policy_discount = 1

        # choose initial action:

        # bandits
        if self.agent_type == 'bandit':
            if pd.isnull(init_action):
                action = np.random.choice(self.actions)
            else:
                action = init_action

        # all other agents
        else:
            if pd.isnull(init_action):
                if pd.isnull(init_state):
                    action = np.random.choice(self.actions)
                else:
                    action = self.choose_action_from_policy(init_state)
            else:
                action = init_action

        return action

    def agent_step(self, last_state, last_action, state, reward, terminal, epsilon, step_size, discount, argmax_action=None):
        # agent step during episode

        # choose action for SARSA, Q-learning, and Expected SARSA
        if self.agent_type in ['q_learning', 'sarsa', 'expected_sarsa']:
            action = self.choose_action_from_policy(state, epsilon=epsilon)
        else:
            action = None

        # calculate value target and estimate
        if self.agent_type == 'bandit':
            # increase arm count
            self.arm_count[last_action] += 1

            # update step size
            step_size = {'value': 1 / self.arm_count[last_action]}

            # get bandit value target and estimate
            target = reward
            estimate = self.values[last_action]

        elif self.agent_type == 'td':
            # get td value target and estimate
            target = self.get_td_target(state, reward, terminal, discount)
            estimate = self.get_value(last_state)

        elif self.agent_type == 'sarsa':
            # get sarsa value target and estimate
            target = self.get_sarsa_target(state, action, reward, terminal, discount)
            estimate = self.get_value(last_state, last_action)

        elif self.agent_type == 'q_learning':
            # get Q-learning value target and estimate
            target, argmax_action = self.get_q_learning_target(state, reward, terminal, discount)
            estimate = self.get_value(last_state, last_action)

        elif self.agent_type == 'expected_sarsa':
            # get expected sarsa target and estimate
            target = self.get_expected_sarsa_target(state, reward, terminal, discount, epsilon)
            estimate = self.get_value(last_state, last_action)

        # update average reward
        if self.use_avg_reward:
            self.avg_reward += step_size['avg_reward'] * (target - estimate)

        # update values
        if self.value_type == 'tabular':
            # tabular case
            self.value_tabular_update(last_state=last_state, last_action=last_action, action=action, step_size=step_size,
                                      discount=discount, estimate=estimate, target=target, argmax_action=argmax_action)
        else:
            # value function approximation
            self.value_approx_update(last_state=last_state, last_action=last_action, action=action, step_size=step_size,
                                     estimate=estimate, target=target, argmax_action=argmax_action)

        # update parameterized policy
        if self.policy_type not in ['tabular', 'softmax']:
            self.policy_update(last_state, last_action, step_size, discount, estimate, target)

        # update beta_m_product and beta_v_product
        if self.value_update_type == 'adam' or self.policy_update_type == 'adam':
            self.beta_m_product *= self.beta_m
            self.beta_v_product *= self.beta_v

        # choose action if we have not done so already
        if pd.isnull(action):
            action = self.choose_action_from_policy(state, epsilon=epsilon)

        return action

    def update_model(self, env, last_state, last_action, state, reward, terminal=False):
        # update the agent's model of the environment

        if last_state not in self.planning_model.keys():
            self.planning_model[last_state] = {}
            self.planning_model[last_state][last_action] = {'total_visits': 1}
            for s in self.states:
                self.planning_model[last_state][last_action][s] = {}
                if s == state:
                    self.planning_model[last_state][last_action][s]['visits'] = 1
                    self.planning_model[last_state][last_action][s]['total_reward'] = reward
                    self.planning_model[last_state][last_action][s]['probability'] = 1
                    self.planning_model[last_state][last_action][s]['reward'] = reward
                    self.planning_model[last_state][last_action][s]['terminal'] = terminal
                else:
                    self.planning_model[last_state][last_action][s]['visits'] = 0
                    self.planning_model[last_state][last_action][s]['total_reward'] = 0
                    self.planning_model[last_state][last_action][s]['probability'] = 0
                    self.planning_model[last_state][last_action][s]['reward'] = env.default_reward
                    self.planning_model[last_state][last_action][s]['terminal'] = terminal

            for action in self.actions:
                if action != last_action:
                    self.planning_model[last_state][action] = {'total_visits': 0}
                    for s in self.states:
                        self.planning_model[last_state][action][s] = {}
                        self.planning_model[last_state][action][s]['visits'] = 0
                        self.planning_model[last_state][action][s]['total_reward'] = 0
                        self.planning_model[last_state][action][s]['reward'] = env.default_reward
                        self.planning_model[last_state][action][s]['terminal'] = terminal

                        if s == last_state:
                            self.planning_model[last_state][action][s]['probability'] = 1
                        else:
                            self.planning_model[last_state][action][s]['probability'] = 0
        else:
            self.planning_model[last_state][last_action]['total_visits'] += 1
            self.planning_model[last_state][last_action][state]['visits'] += 1
            self.planning_model[last_state][last_action][state]['total_reward'] += reward
            self.planning_model[last_state][last_action][state]['reward'] = self.planning_model[last_state][last_action][state]['total_reward'] / self.planning_model[last_state][last_action][state]['visits']
            self.planning_model[last_state][last_action][state]['terminal'] = terminal

            for s in self.states:
                self.planning_model[last_state][last_action][s]['probability'] = self.planning_model[last_state][last_action][s]['visits'] / self.planning_model[last_state][last_action]['total_visits']

    def planning(self, step_size, discount, epsilon, planning_steps):
        # planning using model

        for i in range(planning_steps):
            s = np.random.choice(list(self.planning_model.keys()))
            a = np.random.choice(list(self.planning_model[s].keys()))

            probabilities = []
            for state in self.states:
                probabilities.append(self.planning_model[s][a][state]['probability'])

            s_prime = np.random.choice(self.states, p=probabilities)
            r = self.planning_model[s][a][s_prime]['reward']
            terminal = self.planning_model[s][a][s_prime]['terminal']

            if self.planning_type == 'dyna_plus':
                r += self.planning_kappa * np.sqrt(self.planning_tau[s][a])

            # regular RL update
            _ = self.agent_step(s, a, s_prime, r, terminal, epsilon, step_size, discount)

    def agent_step_with_planning(self, env, last_state, last_action, state, reward, terminal, epsilon, step_size, discount, planning_steps):
        # wrapper for agent step during episode with planning

        # update state visits counts
        if self.planning_type == 'dyna_plus':
            for s in self.states:
                for a in self.actions:
                    self.planning_tau[s][a] += 1
            self.planning_tau[last_state][last_action] = 0

        # regular RL update
        _ = self.agent_step(last_state, last_action, state, reward, terminal, epsilon, step_size, discount)

        # update model
        self.update_model(env, last_state, last_action, state, reward)

        # planning steps
        self.planning(step_size, discount, epsilon, planning_steps)

        # choose action
        action = self.choose_action_from_policy(state, epsilon=epsilon)

        return action

    def agent_step_with_experience_replay(self, last_state, last_action, state, reward, terminal, epsilon, step_size, discount, replay_steps):
        # wrapper for agent step during episode with experience replay

        # append observation to replay buffer
        if len(self.replay_buffer) >= self.replay_buffer_size:
            del self.replay_buffer[0]

        self.replay_buffer.append([last_state, last_action, state, reward, terminal])

        # experience replay:
        if len(self.replay_buffer) > self.replay_buffer_minibatch_size:
            # update target network
            if self.value_type == 'neural_network':
                self.weights_target_value = copy.deepcopy(self.weights_value)
            elif self.value_type in ['nn_pytorch', 'nn_keras']:
                self.neural_network_target_value = copy.deepcopy(self.neural_network_value)

            # perform replay steps
            for i in range(replay_steps):
                # Get sample experiences from the replay buffer
                observation_indices = np.random.choice(np.arange(len(self.replay_buffer)), size=self.replay_buffer_minibatch_size)
                observations = [self.replay_buffer[i] for i in observation_indices]

                # perform experience replay updates
                for observation in observations:
                    _ = self.agent_step(observation[0],  # last state
                                        observation[1],  # last action
                                        observation[2],  # state
                                        observation[3],  # reward
                                        observation[4],  # terminal
                                        epsilon, step_size, discount)

        # choose action
        action = self.choose_action_from_policy(state, epsilon=epsilon)

        return action

    def get_td_target(self, state, reward, terminal, discount):
        # TD(lambda) update
        if not terminal:
            if self.use_avg_reward:
                target = reward - self.avg_reward + self.get_target_value(state)
            else:
                target = reward + discount * self.get_target_value(state)
        else:
            target = reward

        return target

    def get_sarsa_target(self, state, action, reward, terminal, discount):
        # SARSA update
        if not terminal:
            if self.use_avg_reward:
                target = reward - self.avg_reward + self.get_target_value(state, action)
            else:
                target = reward + discount * self.get_target_value(state, action)
        else:
            target = reward

        return target

    def get_q_learning_target(self, state, reward, terminal, discount):
        # Q-Learning update
        if self.value_type == 'tabular':
            values = self.values[state]
        else:
            values = {}
            for a in self.actions:
                values[a] = self.get_target_value(state, a)

        argmax_action = self.argmax(values)

        if not terminal:
            if self.use_avg_reward:
                target = reward - self.avg_reward + values[argmax_action]
            else:
                target = reward + discount * values[argmax_action]
        else:
            target = reward

        return target, argmax_action

    def get_expected_sarsa_target(self, state, reward, terminal, discount, epsilon):
        # Expected SARSA update
        expected_value = 0

        if self.value_type == 'tabular':
            values = self.values[state]
        else:
            values = {}
            for a in self.actions:
                values[a] = self.get_target_value(state, a)

        argmax_action = self.argmax(values)

        if self.policy_type != 'tabular':
            softmax_probs = self.get_softmax_probabilities(state)
        else:
            softmax_probs = None

        for a in self.actions:
            expected_value += self.get_policy(state, a, argmax_action=argmax_action, epsilon=epsilon, softmax_probs=softmax_probs) * values[a]

        if not terminal:
            if self.use_avg_reward:
                target = reward - self.avg_reward + expected_value
            else:
                target = reward + discount * expected_value
        else:
            target = reward

        return target
