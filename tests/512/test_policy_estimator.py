from numpy import random
from exercise_512.generate_agents_and_environments import create_envioronment_agent_and_states
from exercise_512.monte_carlo_offpolicy_estimator import MonteCarloOffPolicyEstimator
from exercise_512.episode_simulator import EpisodeSimulator
import numpy as np


def create_random_q_dict_from_env_and_agent(agent, all_possible_states):
    possible_actions = agent.get_list_of_possible_actions()
    q_dict = {}
    for state in all_possible_states:
        for action in possible_actions:
            state_action_pair = (state, action)
            q_dict[state_action_pair]=random.uniform()
    return q_dict

simple_track, driver, all_possible_states = create_envioronment_agent_and_states()

initial_q_dict = create_random_q_dict_from_env_and_agent(agent=driver,
                                                         all_possible_states=all_possible_states)


policy_estimator = MonteCarloOffPolicyEstimator(agent=driver,
                                                environment=simple_track,
                                                q_dict=initial_q_dict)

def test_policy_estimator():

    bad_policy = policy_estimator.estimate_policy(gama=0.9,
                                                  max_iter=10)

    good_policy = policy_estimator.estimate_policy(gama=0.9,
                                                  max_iter=10)

    print('done with policies')
    ## bad episodes
    driver.set_new_policy(new_policy=bad_policy)
    episode_simulator = EpisodeSimulator(agent=driver, environment=simple_track)
    bad_episode_lens = []
    for episode in range(10000):
        x = episode_simulator.run_episode(start_state=((1, 0), (0, 1)))
        bad_episode_lens.append(len(x[0]))

    print('done with bad simulations')
    ## good episodes
    driver.set_new_policy(new_policy=good_policy)
    episode_simulator = EpisodeSimulator(agent=driver, environment=simple_track)
    good_episode_lens = []
    for episode in range(10000):
        x = episode_simulator.run_episode(start_state=((1, 0), (0, 1)))
        good_episode_lens.append(len(x[0]))

    print('done with good simulations')


    print('bad mean', np.mean(bad_episode_lens))
    print('bad std', np.std(bad_episode_lens))
    print('good mean', np.mean(good_episode_lens))
    print('good std', np.std(good_episode_lens))



