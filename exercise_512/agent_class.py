from numpy import random as random


class DriverAgent:
    def __init__(self, initial_policy,possible_actions, epsilon):
        self.name = 'johnny'
        self.policy_dict = initial_policy
        self.possible_actions = possible_actions
        self.epsilon = epsilon

    def return_action(self, state):
        r = random.uniform()
        if r < self.epsilon:
            return random.choice(a=self.possible_actions)
        else:
            return self.policy_dict[state]

    def change_policy(self, state, new_action):
        self.policy_dict[state] = new_action

    def get_list_of_possible_actions(self):
        return self.possible_actions

    def return_policy_copy(self):
        return self.policy_dict.copy()
