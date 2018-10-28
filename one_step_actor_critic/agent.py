import numpy as np
import itertools
import random
from one_step_actor_critic.aux_functions import softmax


class Agent:
    def __init__(self, possible_actions,  epsilon):
        self.epsilon = epsilon
        self.possible_actions = possible_actions

    def initialize_w(self):
        # get the number of features, so I know the number of w
        n_features = 3
        self.w = [0 for _ in range(n_features)]

    def initialize_th(self):
        # get the number of features, so I know the number of w
        n_features = 3
        self.th = [0 for _ in range(n_features)]

    def return_w(self):
        return self.w


    def state_to_feature_vec(self, state):
        x_dif = 3 - state[0]
        y_dif = 3 - state[1]
        feature_vec = [x_dif, y_dif] + [1]
        return np.array(feature_vec)

    def state_value_estimate(self, state):
        feature_values = self.state_to_feature_vec(state)
        val_estimate = np.dot(a=self.w, b=feature_values)
        return val_estimate

    def state_value_estimate_gradient(self):
        w = self.return_w()
        return np.array(w)



    def state_action_to_feature_vec(self, state, action, environment):
        # pairwise_iterator = itertools.product(list(state), list(action))
        environment.set_state(state)
        next_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
        x_dif = 3 - next_state[0]
        y_dif = 3 - next_state[1]
        feature_vec = [x_dif, y_dif] + [1]
        return np.array(feature_vec)

    def from_state_action_to_q_estimate(self, state, action, environment):
        feature_values = self.state_action_to_feature_vec(state, action, environment=environment)
        val_estimate = np.dot(a=self.w, b=feature_values)
        return val_estimate

    def get_state_action_values(self, state_to_interrogate, environment):
        value_list = []
        possible_actions = self.possible_actions
        random.shuffle(possible_actions)
        for action in possible_actions:
            val_estimate = self.from_state_action_to_q_estimate(state=state_to_interrogate,
                                                                action=action,
                                                                environment=environment)
            value_list.append(val_estimate)
        return value_list

    def get_action_from_policy(self, state_to_interrogate, environment):
        value_list = self.get_state_action_values(state_to_interrogate, environment)
        possible_actions = self.possible_actions
        soft_values = softmax(value_list)
        action_indx_list = list(range(len(possible_actions)))
        chosen_indx = np.random.choice(a=action_indx_list, p=soft_values)
        chosen_action = possible_actions[chosen_indx]
        return chosen_action


    def ln_policy_gradient(self, state, action, environment):
        x = self.state_action_to_feature_vec(state, action, environment)
        sm = softmax(x)
        out = x-sm
        return out


    def return_action(self, state, environment):
        r = np.random.uniform()
        if r < self.epsilon:
            return self.possible_actions[np.random.choice(len(self.possible_actions))]
        else:
            return self.get_action_from_policy(state_to_interrogate=state, environment=environment)

    def update_w(self, new_w):
        assert len(self.w) == len(new_w)
        self.w = new_w

    def increment_w(self, w_increment):
        assert len(self.w) == len(w_increment)
        self.w += w_increment

    def increment_th(self, th_increment):
        assert len(self.w) == len(th_increment)
        self.th += th_increment



