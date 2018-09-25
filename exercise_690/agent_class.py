from numpy import random
from numpy import argmax

class Agent:
    def __init__(self, possible_actions, initial_q,  epsilon):
        self.q_dict = initial_q
        self.epsilon = epsilon
        self.possible_actions = possible_actions

    def get_best_action_for_state_from_Q(self,  state_to_interrogate):
        action_list=[]
        value_list = []
        for state_action, value in self.q_dict.items():
            state = state_action[0]
            if state == state_to_interrogate:
                action = state_action[1]
                action_list.append(action)
                value_list.append(value)
        arg_max = argmax(value_list)
        return action_list[arg_max]

    def return_action(self, state):
        r = random.uniform()
        if r < self.epsilon:
            return self.possible_actions[random.choice(len(self.possible_actions))]
        else:
            return self.get_best_action_for_state_from_Q(state_to_interrogate=state)


    def replace_q_value(self, state_action_to_update, value):
        self.q_dict[state_action_to_update] = value


    def increment_q_value(self, state_action_to_update, value_to_increment):
        self.q_dict[state_action_to_update] += value_to_increment

