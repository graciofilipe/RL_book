#Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the
#difficulties that sample-average methods have for nonstationary problems. Use a modified
#version of the 10-armed testbed in which all the q ⇤ (a) start out equal and then take
#independent random walks (say by adding a normally distributed increment with mean
#zero and standard deviation 0.01 to all the q ⇤ (a) on each step). Prepare plots like
#Figure 2.2 for an action-value method using sample averages, incrementally computed,
#and another action-value method using a constant step-size parameter, ↵ = 0.1. Use
#" = 0.1 and longer runs, say of 10,000 steps.

from exercise_25.bandit_class import SingleArmBandit, KArmBandit
from exercise_25.player_class import KBanditsPlayer
from tqdm import tqdm
import numpy as np

bandit_rewards = [1,2,3,4,3,2,1]
bandit_vars = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
bandit_drifts = [0.01,0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
n_bandits = len(bandit_rewards)

k_arm_bandit = KArmBandit(SingleArmBandit,
                          bandit_rewards=bandit_rewards,
                          bandit_vars=bandit_vars,
                          bandit_drifts=bandit_drifts)


player_ucb = KBanditsPlayer(n_bandits=n_bandits,
                        epsilon=0.01,
                        c = 1,
                        estimate=5,
                        action_selector='ucb')

player_eps = KBanditsPlayer(n_bandits=n_bandits,
                        epsilon=1/16,
                        c = 1,
                        estimate=5,
                        action_selector='epsilon_greedy')


for player in [player_eps, player_ucb]:
    for simulation in tqdm(range(1000)):
        average_reward = []
        reward_list = []
        for _ in range(1000):

            #restart the bandit
            k_arm_bandit = KArmBandit(SingleArmBandit,
                                      bandit_rewards=bandit_rewards,
                                      bandit_vars=bandit_vars,
                                      bandit_drifts=bandit_drifts)


            action = player.return_action_to_take()
            reward = k_arm_bandit.play_action_a(a=action)
            player.collect_reward_and_update_q(reward=reward, action_taken=action)
            reward_list.append(reward)
        average_reward.append(np.average(reward_list))
    print('result', np.average(average_reward))