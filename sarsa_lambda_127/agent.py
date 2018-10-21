import numpy as np
import itertools
import random


class Agent:
    def __init__(self, possible_actions,  epsilon):
        self.epsilon = epsilon
        self.possible_actions = possible_actions
        self.dict_to_w_indx_mapping = {(0, 0): 0,
                                       (1, 0): 1,
                                       (0, 1): 2,
                                       (1, 1): 3,
                                       (2, 0): 4,
                                       (0, 2): 5,
                                       (2, 1): 6,
                                       (1, 2): 7,
                                       (2, 2): 8,
                                       (3, 0): 9,
                                       (0, 3): 10,
                                       (1, 3): 11,
                                       (3, 1): 12,
                                       (3, 2): 13,
                                       (2, 3): 14,
                                       (3, 3): 15}

    def initialize_w(self, sample_state, environment):
        sample_action = self.possible_actions[0]
        # get the number of features, so I know the number of w
        n_features = len(self.dict_to_w_indx_mapping)
        self.w = [0 for _ in range(n_features)]

    def return_w(self):
        return self.w

    def state_action_to_feature_vec(self, state, action, environment):
        # pairwise_iterator = itertools.product(list(state), list(action))
        environment.set_state(state)
        next_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
        # feature_vec = [1] # offset
        n_features = len(self.dict_to_w_indx_mapping)
        feature_vec = [0 for _ in range(n_features)]
        feature_vec[self.dict_to_w_indx_mapping[next_state]] = 1
        return np.array(feature_vec)

    def from_state_action_to_q_estimate(self, state, action, environment):
        feature_values = self.state_action_to_feature_vec(state, action, environment=environment)
        val_estimate = np.dot(a=self.w, b=feature_values)
        return val_estimate

    def get_best_action_for_state_from_Q(self, state_to_interrogate, environment):
        value_list = []
        possible_actions = self.possible_actions
        random.shuffle(possible_actions)
        for action in possible_actions:
            val_estimate = self.from_state_action_to_q_estimate(state=state_to_interrogate,
                                                                action=action,
                                                                environment=environment)
            value_list.append(val_estimate)
        arg_max = np.argmax(value_list)
        x = possible_actions[arg_max]
        return x

    def return_action(self, state, environment):
        r = np.random.uniform()
        if r < self.epsilon:
            return self.possible_actions[np.random.choice(len(self.possible_actions))]
        else:
            return self.get_best_action_for_state_from_Q(state_to_interrogate=state, environment=environment)

    def update_w(self, new_w):
        assert len(self.w) == len(new_w)
        self.w = new_w

