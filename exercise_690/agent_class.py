class Agent:
    def __init__(self, possible_actions, initial_q, initial_policy):
        self.q_dict = initial_q
        self.policy = initial_policy

    def return_action(self, state):
        return self.policy[state]
