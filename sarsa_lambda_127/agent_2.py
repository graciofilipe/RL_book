import numpy as np
import itertools
import random
from sarsa_lambda_127.agent import Agent


class Agent2(Agent):
    def __init__(self, possible_actions,  epsilon):
        Agent.__init__(self, possible_actions, epsilon)
        self.epsilon = epsilon
        self.possible_actions = possible_actions


    def initialize_w(self, sample_state, environment):
        sample_action = self.possible_actions[0]
        # get the number of features, so I know the number of w
        n_features = 3
        self.w = [0 for _ in range(n_features)]

    def state_action_to_feature_vec(self, state, action, environment):
        # pairwise_iterator = itertools.product(list(state), list(action))
        environment.set_state(state)
        next_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
        x_dif = 3 - next_state[0]
        y_dif = 3 - next_state[1]
        feature_vec = [x_dif, y_dif] + [1]
        return np.array(feature_vec)
