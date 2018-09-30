from exercise_690.generate_agents_and_environments import create_environment_agent_and_states
from exercise_690.sarsa import Sarsa
from exercise_690.episode_simulator_class import EpisodeSimulator
import numpy as np


def test_just_load_things():

    environment, agent = create_environment_agent_and_states(grid_shape=(5, 5),
                                                             initial_position=(0, 2),
                                                             final_position=(2, 4),
                                                             epsilon=0.1)
    sarsa_simulator = Sarsa(alpha=0.5, gama=0.9)


    e, a = sarsa_simulator.run_sarsa(environment=environment, agent=agent,
                                     initial_state=(0,2), n_episodes=10)


def test_episode_simulator_with_print():
    environment, agent = create_environment_agent_and_states(grid_shape=(5, 5),
                                                             initial_position=(0, 2),
                                                             final_position=(2, 4),
                                                             epsilon=0.2)
    sarsa_simulator = Sarsa(alpha=0.5, gama=0.9)
    environment, agent = sarsa_simulator.run_sarsa(environment=environment,
                                                   agent=agent,
                                                   initial_state=(0, 2),
                                                   n_episodes=10)

    episode_simulator = EpisodeSimulator(agent=agent, environment=environment)
    state_action_tuple_list, reward_list = episode_simulator.run_episode(start_state=(0, 2))
    for i in range(len(state_action_tuple_list)):
        print('\n')
        print('state', state_action_tuple_list[i][0])
        print('action', state_action_tuple_list[i][1])
        print('reward', reward_list[i])



def test_sarsa_creates_better_policy():
    n_sims = 1000
    sarsa_episodes = 1000
    grid_shape = (5, 5)
    final_position = (4, 2)
    initial_position = (0, 2 )


    ## bad
    environment_bad, agent_bad = create_environment_agent_and_states(grid_shape=grid_shape,
                                                                     initial_position=initial_position,
                                                                     final_position=final_position,
                                                                     epsilon=0.1)
    sarsa_simulator_bad = Sarsa(alpha=0.5, gama=0.9)
    environment_bad, agent_bad = sarsa_simulator_bad.run_sarsa(environment=environment_bad,
                                                               agent=agent_bad,
                                                               initial_state=initial_position,
                                                               n_episodes=10)
    bad_list = []
    for i in range(n_sims):
        episode_simulator = EpisodeSimulator(agent=agent_bad, environment=environment_bad)
        state_action_tuple_list_bad, reward_list_bad = episode_simulator.run_episode(start_state=initial_position)
        bad_list.append(sum(reward_list_bad))



    ## good
    environment_good, agent_good = create_environment_agent_and_states(grid_shape=grid_shape,
                                                                       initial_position=initial_position,
                                                                       final_position=final_position,
                                                                       epsilon=0.1)
    sarsa_simulator_good = Sarsa(alpha=0.5, gama=0.9)
    environment_good, agent_good = sarsa_simulator_good.run_sarsa(environment=environment_good,
                                                                  agent=agent_good,
                                                                  initial_state=initial_position,
                                                                  n_episodes=sarsa_episodes)
    good_list = []
    for i in range(n_sims):
        episode_simulator = EpisodeSimulator(agent=agent_good, environment=environment_good)
        state_action_tuple_list_good, reward_list_good = episode_simulator.run_episode(start_state=initial_position)
        good_list.append(sum(reward_list_good))
    
    print('bad mean', np.mean(bad_list))
    print('bad sd', np.std(bad_list))
    print('good mean', np.mean(good_list))
    print('good sd', np.std(good_list))
    print(state_action_tuple_list_good)
    print(agent_good.q_dict)

