import numpy.random as random
import numpy as np

class KBanditsPlayer:

    def __init__(self, n_bandits, epsilon, c, estimate, action_selector):
        self.n_bandits = n_bandits
        self.actions = [i for i in range(n_bandits)]
        self.q_estimates = {action:estimate for action in self.actions}
        self.epsilon = epsilon
        self.c = c
        self.number_of_actions = 1
        self.action_selector = action_selector
        self.number_of_actions_details = {action:1 for action in self.actions}




    def epsilon_greedy_action_selector(self):
        r = random.uniform()
        if r < self.epsilon:
            action_to_take = random.choice(self.actions)
        else:
            action_to_take = max(self.q_estimates, key=self.q_estimates.get)
        self.number_of_actions += 1

        return action_to_take

    def ucb_action_selector(self):

        ln_t = np.log(self.number_of_actions)
        ucb = [self.q_estimates[action] + \
               self.c*np.sqrt(ln_t/self.number_of_actions_details[action]) \
               for action in self.actions]

        action_to_take = np.argmax(ucb)
        self.number_of_actions += 1
        self.number_of_actions_details[action_to_take]  += 1

        return action_to_take


    def return_action_to_take(self):
        if self.action_selector == 'epsilon_greedy':
            return self.epsilon_greedy_action_selector()
        if self.action_selector == 'ucb':
            return self.ucb_action_selector()



    def collect_reward_and_update_q(self, reward, action_taken):
        update_rate = 1/self.number_of_actions
        self.q_estimates[action_taken] += update_rate*(reward - self.q_estimates[action_taken])


    def print_current_q_estimates(self):
        for action, value in self.q_estimates.items():
            print('action:', action, ' - value:', round(value, 2))
