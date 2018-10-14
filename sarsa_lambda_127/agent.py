import numpy as np
import itertools


class Agent:
    def __init__(self, possible_actions,  epsilon):
        self.epsilon = epsilon
        self.possible_actions = possible_actions

    def initialize_w(self, sample_state):
        sample_action = self.possible_actions[0]
        # get the number of features, so I know the number of w
        feature_vec = self.state_action_to_feature_vec(state = sample_state, action=sample_action)
        n_features = len(feature_vec)
        self.w = [np.random.random() for _ in range(n_features)]

    def return_w(self):
        return self.w


    def state_action_to_feature_vec(self, state, action):
        state_action = list(state) + list(action)
        pairwise_iterator = itertools.product(list(state), list(action))
        state_action_products = [x[0]*x[1] for x in pairwise_iterator]
        pairwise_iterator = itertools.product(list(state), list(action))
        state_action_differences = [x[0]-x[1] for x in pairwise_iterator]
        feature_vec = state_action + state_action_products + state_action_differences
        return feature_vec

    def from_state_action_to_q_estimate(self, state, action):
        feature_values = self.state_action_to_feature_vec(state, action)
        val_estimate = np.dot(a=self.w, b=feature_values)
        return val_estimate

    def get_best_action_for_state_from_Q(self,  state_to_interrogate):
        value_list = []
        for action in self.possible_actions:
            val_estimate = self.from_state_action_to_q_estimate(state=state_to_interrogate,
                                                 action=action)
            value_list.append(val_estimate)
        arg_max = np.argmax(value_list)
        return self.possible_actions[arg_max]

    def return_action(self, state):
        r = np.random.uniform()
        if r < self.epsilon:
            return self.possible_actions[np.random.choice(len(self.possible_actions))]
        else:
            return self.get_best_action_for_state_from_Q(state_to_interrogate=state)

    def update_w(self, new_w):
        assert len(self.w) == len(new_w)
        self.w = new_w

