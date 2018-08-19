import numpy.random as random


class SingleArmBandit:

    def __init__(self, reward_mean, reward_var, movement_var):
        self.reward_mean = reward_mean
        self.reward_var = reward_var
        self.movement_var = movement_var

    def update_reward_mean(self):
        r = random.normal(0, self.movement_var)
        self.reward_mean += r

    def get_reward(self):
        r = random.normal(self.reward_mean, self.reward_var)
        self.update_reward_mean()
        return r


class KArmBandit:

    def __init__(self, bandit_class, bandit_rewards, bandit_vars, bandit_drifts):
        n_bandits = len(bandit_rewards)
        self.bandit_list = \
            [bandit_class(bandit_rewards[i], bandit_vars[i], bandit_drifts[i]) \
             for i in range(n_bandits)]

    def play_action_a(self, a):
        bandit_in_play = self.bandit_list[a]
        return bandit_in_play.get_reward()