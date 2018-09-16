class MonteCarloOffPolicyEstimator:
    def __init__(self, agent, environment, q_dict,  episode_simulator):
        self.agent = agent
        self.environment = environment
        self.episode_simulator = episode_simulator
        self.state_action_counter = {state: 0 for state in q_dict.keys()}
        self.q_dict = q_dict
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

    def estimate_policy(self, gama, max_iter):
        for _ in range(max_iter):
            behavior_policy = self.policy.copy()

            # generate episode


