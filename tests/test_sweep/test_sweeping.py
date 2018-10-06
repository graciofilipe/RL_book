from prioritised_sweep_8.generate_agent_and_eng import create_environment_agent_and_states
from prioritised_sweep_8.sweeper import run_sweep, add_to_priority_list
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

    good_iters = 15000
    n_sims = 500
    n_runs = 2

    good_episode_lens = []
    bad_episode_lens = []

    for run in range(n_runs):

        environment, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                                 initial_position=(0, 0),
                                                                 final_position=(3, 3),
                                                                 epsilon=0.1)
        bad_agent = run_sweep(environment=environment,
                              agent=agent,
                              alpha=0.5,
                              gama=0.9,
                              n_iter=1,
                              theta=0.1)
        for i in range(n_sims):
            state_action_tuple_list, reward_list = run_episode(start_state=(0, 0),
                                                               environment=environment,
                                                               agent=bad_agent)
            episode_len = len(reward_list)
            bad_episode_lens.append(episode_len)



        #####################################
        environment, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                                 initial_position=(0, 0),
                                                                 final_position=(3, 3),
                                                                 epsilon=0.1)
        good_agent = run_sweep(environment=environment,
                               agent=agent,
                              alpha=0.5,
                              gama=0.9,
                              n_iter=good_iters,
                              theta=0.1)
        for i in range(n_sims):
            state_action_tuple_list, reward_list = run_episode(start_state=(0, 0),
                                                               environment=environment,
                                                               agent=good_agent)
            episode_len = len(reward_list)
            good_episode_lens.append(episode_len)

    print('\n')
    print('good_episode_lens mean', np.mean(good_episode_lens), 'and sd', np.std(good_episode_lens))
    print('bad_episode_lens mean', np.mean(bad_episode_lens), 'and sd', np.std(bad_episode_lens))



def test_item_is_added_to_priority_list():
    p_list = [(1, (1, 2)), (3, (0, 1))]
    new_item = (2, (1, 3))
    new_list = add_to_priority_list(current_priority_list=p_list, item_to_add=new_item)
    assert new_list == [(1, (1, 2)), (2, (1, 3)), (3, (0, 1))]


def test_item_isnt_added_to_priority_list():
    p_list = [(3, (1, 2)), (1, (0, 1))]
    new_item = (2, (1, 2))
    new_list = add_to_priority_list(current_priority_list=p_list, item_to_add=new_item)
    assert new_list == [(1, (0, 1)), (3, (1, 2))]


def test_item_is_replaced_in_priority_list():
    p_list = [(3, (1, 2)), (1, (0, 1))]
    new_item = (4, (1, 2))
    new_list = add_to_priority_list(current_priority_list=p_list, item_to_add=new_item)
    assert new_list == [(1, (0, 1)), (4, (1, 2))]

def test_item_is_added_to_empty_list():
    p_list = []
    new_item = (4, (1, 2))
    new_list = add_to_priority_list(current_priority_list=p_list, item_to_add=new_item)
    assert new_list == [(4, (1, 2))]