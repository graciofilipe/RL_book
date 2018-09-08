class CarBusinessAgent:
    def __init__(self, possible_actions):
        self.policy_dict = {(i, j):0
                       for i in range(0, 33)
                       for j in range(0, 33)}
        self.total_reward = 0
        self.possible_actions = possible_actions


    def return_action(self, state):
        return self.policy_dict[state]

    def change_policy(self, state, new_action):
        self.policy_dict[state]=new_action

    def get_list_of_possible_actions(self):
        return self.possible_actions

    def return_policy_copy(self):
        return self.policy_dict.copy()
