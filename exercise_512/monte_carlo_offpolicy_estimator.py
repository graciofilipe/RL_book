from numpy import random
from exercise_512.episode_simulator import EpisodeSimulator

class MonteCarloOffPolicyEstimator:
    def __init__(self, agent, environment, q_dict, gama):
        self.agent = agent
        self.environment = environment
        self.episode_simulator = episode_simulator
        self.state_action_counter = {state: 0 for state in q_dict.keys()}
        self.q_dict = q_dict
        self.gama = gama
        self.policy = {}
        self.infer_policy_from_q()

        # infer policy from the q_function
    def infer_policy_from_q(self):
        state_val = {}
        for state_action_tuple, q_val in self.q_dict.items():
            state, action = state_action_tuple[0], state_action_tuple[1]
            if q_val > state_val.get(state, 0):
                self.policy[state] = action
                state_val[state] = q_val

    def get_best_action_for_state_from_Q(self, ):

    def estimate_policy(self, gama, max_iter):
        for _ in range(max_iter):
            behavior_policy = self.policy.copy()
            self.agent.set_new_policy(new_policy = behavior_policy)

            g = 0
            w = 1

            # generate episode
            episode_simulator = EpisodeSimulator(agent=self.agent, environment=self.environment)
            random_start_state = self.environment.starting_locations[random.choice(len(self.environment.starting_locations))]
            state_action, rewards = episode_simulator.run_episode(start_state=random_start_state)
            episode_len = len(state_action)
            for t in range(episode_len-1, -1, -1):
                g = g*self.gama* + rewards[t]
                c_dict[state_action[t]] = c_dict.get(state_action, 0) + w
                q_dict[state_action[t]] += (w/c_dict[state_action[t]])*(g - q_dict[state_action[t]])







