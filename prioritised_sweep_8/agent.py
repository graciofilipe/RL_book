from numpy import argmax
from numpy.random import random
from numpy.random import uniform
from numpy.random import choice


class SweepingAgent:
    def __init__(self, q_initial, epsilon, possible_actions, model):
        self.q_dict = q_initial
        self.epsilon = epsilon
        self.possible_actions = possible_actions
        self.model = model

    def update_model(self, state_action, reward_newstate):
        self.model[state_action] = reward_newstate

    def return_model_based_state_action_reward_response(self, state_action):
        return self.model[state_action]

    def get_best_action_for_state_from_Q(self, state_to_interrogate):
        action_list = []
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
        r = uniform()
        if r < self.epsilon:
            return self.possible_actions[choice(len(self.possible_actions))]
        else:
            return self.get_best_action_for_state_from_Q(state_to_interrogate=state)


    def replace_q_value(self, state_action_to_update, value):
        self.q_dict[state_action_to_update] = value


    def increment_q_value(self, state_action_to_update, value_to_increment):
        self.q_dict[state_action_to_update] += value_to_increment


    def return_state_action_leading_to(self, state_to_query):
        preceding_state_actions_list = []
        rewards_list = []
        for state_action, reward_state in self.model.items():
            end_state = reward_state[1]
            reward = reward_state[0]
            if state_to_query == end_state:
                preceding_state_actions_list.append(state_action)
                rewards_list.append(reward)
        return preceding_state_actions_list, rewards_list
