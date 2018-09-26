from exercise_690.generate_agents_and_environments import create_environment_agent_and_states
from exercise_690.sarsa import Sarsa
from exercise_690.episode_simulator_class import EpisodeSimulator



def test_just_load_things():

    environment, agent = create_environment_agent_and_states(grid_shape=(5, 5),
                                                             initial_position=(0, 2),
                                                             final_position=(4, 2),
                                                             epsilon=0.1)
    sarsa_simulator = Sarsa(alpha=0.5, gama=0.9)


    e, a = sarsa_simulator.run_sarsa(environment=environment, agent=agent,
                                     initial_state=(0,2), n_episodes=10)


def test_episode_simulator_with_print():
    environment, agent = create_environment_agent_and_states(grid_shape=(5, 5),
                                                             initial_position=(0, 2),
                                                             final_position=(4, 2),
                                                             epsilon=0.2)
    sarsa_simulator = Sarsa(alpha=0.5, gama=0.9)
    environment, agent = sarsa_simulator.run_sarsa(environment=environment,
                                                   agent=agent,
                                                   initial_state=(0, 2),
                                                   n_episodes=10)
    episode_simulator = EpisodeSimulator(agent=agent, environment=environment)
    state_action_tuple_list, reward_list = episode_simulator.run_episode(start_state=(0,2))
    for i in range(len(state_action_tuple_list)):
        print('\n')
        print('state', state_action_tuple_list[i][0])
        print('action', state_action_tuple_list[i][1])
        print('reward', reward_list[i])
