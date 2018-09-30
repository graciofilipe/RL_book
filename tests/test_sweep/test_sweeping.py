from prioritised_sweep_8.generate_agent_and_eng import create_environment_agent_and_states
from prioritised_sweep_8.sweeper import run_sweep
from prioritised_sweep_8.episode_simulator_class import run_episode
import numpy as np


def test_basic_run_through():

    environment, agent = create_environment_agent_and_states(grid_shape=(6, 6),
                                                             initial_position=(0, 0),
                                                             final_position=(5, 5),
                                                             epsilon=0.1)

    new_agent = run_sweep(environment=environment,
                          agent=agent,
                          alpha=0.5,
                          gama=0.9,
                          n_iter=3,
                          theta=0.1)

def test_episode_simulator():
    environment, agent = create_environment_agent_and_states(grid_shape=(6, 6),
                                                             initial_position=(0, 0),
                                                             final_position=(5, 5),
                                                             epsilon=0.1)

    state_action_tuple_list, reward_list = run_episode(start_state=(0,0),
                                                       environment=environment,
                                                       agent=agent)
    episode_len = len(reward_list)
    print('episode_len', episode_len)




def test_if_more_iterations_produce_better_episodes():

    good_iters = 30000
    n_sims = 1000

    environment, agent = create_environment_agent_and_states(grid_shape=(6, 6),
                                                             initial_position=(0, 0),
                                                             final_position=(5, 5),
                                                             epsilon=0.1)
    bad_agent = run_sweep(environment=environment,
                          agent=agent,
                          alpha=0.5,
                          gama=0.9,
                          n_iter=1,
                          theta=0.1)
    bad_episode_lens = []
    for i in range(n_sims):
        state_action_tuple_list, reward_list = run_episode(start_state=(0, 0),
                                                           environment=environment,
                                                           agent=bad_agent)
        episode_len = len(reward_list)
        bad_episode_lens.append(episode_len)






    environment, agent = create_environment_agent_and_states(grid_shape=(6, 6),
                                                             initial_position=(0, 0),
                                                             final_position=(5, 5),
                                                             epsilon=0.1)
    good_agent = run_sweep(environment=environment,
                           agent=agent,
                          alpha=0.5,
                          gama=0.9,
                          n_iter=good_iters,
                          theta=0.1)
    good_episode_lens = []
    for i in range(n_sims):
        state_action_tuple_list, reward_list = run_episode(start_state=(0, 0),
                                                           environment=environment,
                                                           agent=good_agent)
        episode_len = len(reward_list)
        good_episode_lens.append(episode_len)

    print('\n')
    print('good_episode_lens mean', np.mean(good_episode_lens), 'and sd', np.std(good_episode_lens))
    print('bad_episode_lens mean', np.mean(bad_episode_lens), 'and sd', np.std(bad_episode_lens))





