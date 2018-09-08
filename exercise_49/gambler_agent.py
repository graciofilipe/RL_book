class GamblerAgent:
    def __init__(self, initial_money, initial_policy, possible_actions):
        self.capital = initial_money
        self.policy_dict = initial_policy
        self.possible_actions = possible_actions

    def return_action(self, state):
        return self.policy_dict[state]

    def change_policy(self, state, new_action):
        self.policy_dict[state]=new_action

    def get_list_of_possible_actions(self):
        return self.possible_actions

    def return_policy_copy(self):
        return self.policy_dict.copy()