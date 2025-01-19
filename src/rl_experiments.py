import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

from src.rl_main import ReinforcementLearning


class RLExperiments:
    def __init__(self):
        self.rl = None

    def start_experiment(self, agent, env, state_representation=None):
        self.rl = ReinforcementLearning(agent, env, state_representation)

    def bandit(self, agent, env, num_episodes, num_steps, step_size, epsilon):
        average_best = 0
        self.start_experiment(agent, env)
        for episode in tqdm(range(num_episodes)):
            np.random.seed(episode)

            # start episode
            last_state, last_action = self.rl.rl_start()

            average_best += np.max(self.rl.environment.arms)

            for i in range(num_steps):
                # the environment and agent take a step and return the reward and action taken
                reward, state, action, _ = self.rl.rl_step(last_state, last_action, step_size=step_size, epsilon=epsilon)

                last_state = state
                last_action = action

        # get experiment data
        results_df = self.rl.get_data()

        # plot results
        average_rewards = []
        for step in range(1, num_steps + 1):
            average_rewards.append(results_df[results_df['step'] == step]['reward'].mean())

        plot_df = pd.DataFrame()
        plot_df['steps'] = np.arange(1, num_steps + 1)
        plot_df['best_possible'] = [average_best / num_episodes for _ in range(num_steps)]
        plot_df['average_reward'] = average_rewards

        fig = go.Figure()

        fig.add_trace(go.Scatter(mode='lines', name='Best Possible', x=plot_df['steps'], y=plot_df['best_possible'], line=dict(dash='dash')))
        fig.add_trace(go.Scatter(mode='lines', name='RL Agent', x=plot_df['steps'], y=plot_df['average_reward']))

        fig.update_xaxes(title='Steps')
        fig.update_yaxes(title='Average Reward')
        fig.update_layout(template='plotly_white')

        fig.show()

        return results_df, fig

    def policy_evaluation(self, agent, env, discount, theta):
        self.start_experiment(agent, env)

        self.rl.agent.dp_policy_evaluation(self.rl.environment, discount, theta)

        values = []
        for state in self.rl.agent.states:
            values.append(self.rl.agent.get_value(state))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=self.rl.agent.states, y=values))

        fig.update_xaxes(title='State')
        fig.update_yaxes(title='Value')
        fig.update_layout(template='plotly_white')

        fig.show()

        return fig

    def policy_iteration(self, agent, env, discount, theta):
        self.start_experiment(agent, env)

        self.rl.agent.dp_policy_iteration(self.rl.environment, discount, theta)

        z = []
        for state in self.rl.agent.states:
            z.append([])
            for action in self.rl.agent.actions:
                z[-1].append(self.rl.agent.get_policy(state, action))

        fig = go.Figure(data=go.Heatmap(z=z, colorscale='blues', colorbar=dict(title='Policy (Probability)')))
        fig.update_xaxes(title='Action', dtick=1)
        fig.update_yaxes(title='State', dtick=1)
        fig.show()

        return fig

    def value_iteration(self, agent, env, discount, theta):
        self.start_experiment(agent, env)

        self.rl.agent.dp_value_iteration(self.rl.environment, discount, theta)

        z = []
        for state in self.rl.agent.states:
            z.append([])
            for action in self.rl.agent.actions:
                z[-1].append(self.rl.agent.get_policy(state, action))

        fig = go.Figure(data=go.Heatmap(z=z, colorscale='blues', colorbar=dict(title='Policy (Probability)')))
        fig.update_xaxes(title='Action', dtick=1)
        fig.update_yaxes(title='State', dtick=1)
        fig.show()

        return fig

    def td_lambda(self, agent, env, num_episodes, step_size, discount):
        self.start_experiment(agent, env)

        for episode in tqdm(range(1, num_episodes + 1)):
            np.random.seed(episode)

            terminal = False
            last_state, last_action = self.rl.rl_start()
            while not terminal:
                reward, state, action, terminal = self.rl.rl_step(last_state, last_action, step_size=step_size, discount=discount)

                last_state = state
                last_action = action

        # get experiment data
        results_df = self.rl.get_data()

        # plot results
        x = []
        y = []
        z = []
        for h in range(self.rl.environment.grid_h):
            x.append(h)
            z.append([])
            for w in range(self.rl.environment.grid_w):
                y.append(w)
                z[-1].append(self.rl.agent.get_value(self.rl.environment.get_state((w, h))))

        print('Start State: bottom left corner')
        print('Goal State: bottom right corner')
        print('Cliff: bottom excluding start/goal states')

        fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale='viridis', colorbar=dict(title='Value')))

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        fig.show()

        return results_df, fig

    def q_learning_sarsa(self, agent, env, num_runs, num_episodes, epsilon, step_size, discount):
        all_results = pd.DataFrame()
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env)
            for episode in range(1, num_episodes + 1):
                np.random.seed(run+episode)

                terminal = False
                last_state, last_action = self.rl.rl_start()
                while not terminal:
                    reward, state, action, terminal = self.rl.rl_step(last_state, last_action, epsilon=epsilon, step_size=step_size, discount=discount)

                    last_state = state
                    last_action = action

            # get experiment data
            results_df = self.rl.get_data()

            # filter to last 10 episodes
            results_df = results_df[results_df['episode'] > num_episodes - 10]

            # add to all results
            results_df['run'] = run + 1
            all_results = pd.concat([all_results, results_df], ignore_index=True)

        # get state visits in last 10 episodes
        state_visits = {}
        for state in self.rl.agent.states:
            state_visits[state] = np.max([len(all_results[all_results['state'] == state]) / num_runs,
                                          len(all_results[all_results['next_state'] == state]) / num_runs])

        # plot results
        print('Start State: bottom left corner')
        print('Goal State: bottom right corner')
        print('Cliff: bottom excluding start/goal states')

        x = []
        y = []
        z = []
        for h in range(self.rl.environment.grid_h):
            x.append(h)
            z.append([])
            for w in range(self.rl.environment.grid_w):
                y.append(w)
                z[-1].append(state_visits[self.rl.environment.get_state((w, h))])

        fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale='viridis', colorbar=dict(title='State Visits During last 10 Episodes')))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()

        return all_results, fig

    def dyna_q_planning_num_steps(self, agent, env, num_runs, num_episodes, epsilon, step_size, discount, planning_steps_list):
        results = {}
        for planning_steps in planning_steps_list:
            print('Planning Steps: ' + str(planning_steps))
            results[planning_steps] = pd.DataFrame()
            for run in tqdm(range(num_runs)):
                self.start_experiment(agent, env)
                for episode in range(1, num_episodes + 1):
                    np.random.seed(run+episode)

                    terminal = False
                    last_state, last_action = self.rl.rl_start()
                    while not terminal:
                        reward, state, action, terminal = self.rl.rl_step(last_state, last_action, step_size=step_size,
                                                                          discount=discount, epsilon=epsilon,
                                                                          planning=True, planning_steps=planning_steps)

                        last_state = state
                        last_action = action

                # get experiment data
                results_df = self.rl.get_data()

                # add to all results
                results_df['run'] = run + 1
                results[planning_steps] = pd.concat([results[planning_steps], results_df], ignore_index=True)

        # plot results
        fig = go.Figure()
        plot_df = pd.DataFrame()
        plot_df['episode'] = list(range(1, num_episodes + 1))

        for planning_steps in planning_steps_list:
            df = results[planning_steps]
            episode_results_list = []
            for episode in range(1, num_episodes + 1):
                df_episode = df[df['episode'] == episode][['run', 'step']].groupby('run').count().reset_index()
                episode_results_list.append(df_episode['step'].mean())

            plot_df['planning_steps_' + str(planning_steps)] = episode_results_list

            fig.add_trace(go.Scatter(mode='lines', name='Planning Steps: ' + str(planning_steps),
                                     x=plot_df['episode'], y=plot_df['planning_steps_' + str(planning_steps)]))

        fig.update_xaxes(title='Episode')
        fig.update_yaxes(title='Steps per Episode')
        fig.update_layout(template='plotly_white')

        fig.show()

        return results, fig

    def dyna_q_planning_state_visits(self, agent, env, num_runs, num_episodes, epsilon, step_size, discount, planning_steps, obstacle_switch=True):
        print('Initial Obstacles:')
        env.env_start()
        env.plot_env()

        all_results = pd.DataFrame()
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env)
            self.rl.environment.reset_counter()
            for episode in range(1, num_episodes + 1):
                np.random.seed(run+episode)

                terminal = False
                last_state, last_action = self.rl.rl_start()
                while not terminal:
                    reward, state, action, terminal = self.rl.rl_step(last_state, last_action, step_size=step_size,
                                                                      discount=discount, epsilon=epsilon,
                                                                      planning=True, planning_steps=planning_steps)

                    last_state = state
                    last_action = action

            # get experiment data
            results_df = self.rl.get_data()

            # filter to last 10 episodes
            results_df = results_df[results_df['episode'] > num_episodes - 10]

            # add to all results
            results_df['run'] = run + 1
            all_results = pd.concat([all_results, results_df], ignore_index=True)

        # get state visits in last 10 episodes
        state_visits = {}
        for state in self.rl.agent.states:
            state_visits[state] = np.max([len(all_results[all_results['state'] == state]) / num_runs,
                                          len(all_results[all_results['next_state'] == state]) / num_runs])

        # plot results
        if obstacle_switch:
            print('Final Obstacles:')
            self.rl.environment.plot_env()

        x = []
        y = []
        z = []
        for h in range(self.rl.environment.grid_h):
            x.append(h)
            z.append([])
            for w in range(self.rl.environment.grid_w):
                y.append(w)
                z[-1].append(state_visits[self.rl.environment.get_state((w, h))])

        fig = go.Figure(data=go.Heatmap(x=x, y=y, z=z, colorscale='viridis', colorbar=dict(title='State Visits During last 10 Episodes')))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.show()

        return all_results, fig

    def td_semigradient(self, agent, env, state_representation, num_runs, num_episodes, epsilon, step_size, discount):
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env, state_representation)
            for episode in range(1, num_episodes + 1):
                np.random.seed(run+episode)

                terminal = False
                last_state, last_action = self.rl.rl_start()
                while not terminal:
                    reward, state, action, terminal = self.rl.rl_step(last_state, last_action, epsilon=epsilon,
                                                                      step_size=step_size, discount=discount)

                    last_state = state
                    last_action = action

        # plot results
        states_list = []
        values_list = []
        for state in range(1, self.rl.environment.num_states + 1):
            states_list.append(state)
            values_list.append(self.rl.agent.get_value(self.rl.state_representation(state)))

        fig = go.Figure(data=go.Scatter(mode='lines', x=states_list, y=values_list))

        fig.update_xaxes(title='State')
        fig.update_yaxes(title='State Value')
        fig.update_layout(template='plotly_white')

        fig.show()

        return fig

    def sarsa_semigradient(self, agent, env, state_representation, num_runs, num_episodes, epsilon, step_size, discount):
        all_results = pd.DataFrame()
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env, state_representation)
            for episode in range(1, num_episodes + 1):
                np.random.seed(run+episode)

                terminal = False
                last_state, last_action = self.rl.rl_start()
                while not terminal:
                    reward, state, action, terminal = self.rl.rl_step(last_state, last_action, epsilon=epsilon,
                                                                      step_size=step_size, discount=discount)

                    last_state = state
                    last_action = action

                # get experiment data
                results_df = self.rl.get_data()

                # add to all results
                results_df['run'] = run + 1
                all_results = pd.concat([all_results, results_df], ignore_index=True)

        # plot results
        fig = go.Figure()
        plot_df = pd.DataFrame()
        plot_df['episode'] = list(range(1, num_episodes + 1))

        episode_results_list = []
        for episode in range(1, num_episodes + 1):
            df_episode = all_results[all_results['episode'] == episode][['run', 'step']].groupby('run').count().reset_index()
            episode_results_list.append(df_episode['step'].mean())

        plot_df['results'] = episode_results_list

        fig.add_trace(go.Scatter(mode='lines', x=plot_df['episode'], y=plot_df['results']))

        fig.update_xaxes(title='Episodes')
        fig.update_yaxes(title='Steps per Episode')
        fig.update_layout(template='plotly_white')

        fig.show()

        return all_results, fig

    def actor_critic(self, agent, env, state_representation, num_runs, max_steps, step_size):
        all_results = pd.DataFrame()
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env, state_representation)
            np.random.seed(run)

            num_steps = 0
            last_state, last_action = self.rl.rl_start()
            while num_steps <= max_steps:
                reward, state, action, terminal = self.rl.rl_step(last_state, last_action, step_size=step_size)

                last_state = state
                last_action = action

                num_steps += 1

            # get experiment data
            results_df = self.rl.get_data()

            # add to all results
            results_df['run'] = run + 1
            results_df['total_reward'] = results_df['reward'].cumsum()
            all_results = pd.concat([all_results, results_df], ignore_index=True)

        # plot results
        fig = go.Figure()
        plot_df = pd.DataFrame()
        plot_df['step'] = list(range(1, max_steps + 1))

        step_results_list = []
        for step in range(1, max_steps + 1):
            df_step = all_results[all_results['step'] == step]
            step_results_list.append(df_step['total_reward'].mean())

        plot_df['results'] = step_results_list

        fig.add_trace(go.Scatter(mode='lines', x=plot_df['step'], y=plot_df['results']))

        fig.update_xaxes(title='Training Step')
        fig.update_yaxes(title='Average Total Cost')
        fig.update_layout(template='plotly_white')

        fig.show()

        return all_results, fig

    def dqn_replay_buffer(self, agent, env, num_runs, num_episodes, step_size, discount, experience_replay, experience_replay_steps):
        all_results = pd.DataFrame()
        for run in tqdm(range(num_runs)):
            self.start_experiment(agent, env)
            for episode in range(1, num_episodes + 1):
                np.random.seed(run + episode)

                terminal = False
                last_state, last_action = self.rl.rl_start()
                while not terminal:
                    reward, state, action, terminal = self.rl.rl_step(last_state, last_action,
                                                                      step_size=step_size, discount=discount,
                                                                      experience_replay=experience_replay,
                                                                      experience_replay_steps=experience_replay_steps)

                    last_state = state
                    last_action = action

                # get experiment data
                results_df = self.rl.get_data()

                # add to all results
                results_df['run'] = run + 1
                all_results = pd.concat([all_results, results_df], ignore_index=True)

        # plot results
        fig = go.Figure()
        plot_df = pd.DataFrame()
        plot_df['episode'] = list(range(1, num_episodes + 1))

        episode_results_list = []
        for episode in range(1, num_episodes + 1):
            df_episode = all_results[all_results['episode'] == episode]
            episode_results_list.append(df_episode['reward'].sum())

        plot_df['results'] = episode_results_list

        fig.add_trace(go.Scatter(mode='lines', x=plot_df['episode'], y=plot_df['results']))

        fig.update_xaxes(title='Episodes')
        fig.update_yaxes(title='Total Reward per Episode')
        fig.update_layout(template='plotly_white')

        fig.show()

        return all_results, fig
