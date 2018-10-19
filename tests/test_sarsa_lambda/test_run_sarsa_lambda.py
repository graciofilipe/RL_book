from sarsa_lambda_127.run_sarsa_lambda import run_sarsa_lambda
from sarsa_lambda_127.generate_agent_and_environment import create_environment_agent_and_states
from sarsa_lambda_127.sarsa_lambda_episode_simulator import run_episode

def test_run_sarsa_lambda_no_crash():
    grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                      initial_position=(0,0),
                                                      final_position=(3,3),
                                                      blocks=[],
                                                      epsilon=0.1)

    agent = run_sarsa_lambda(alpha=0.1,
                     lamb=0.5,
                     agent=agent,
                     environment=grid,
                     n_episodes=11,
                     gama = 0.1,
                     state_0=(0, 0))

    print('\n')
    print('agent w', agent.return_w())


def test_run_sarsa_lambda_kinda_works():

    grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                      initial_position=(0,0),
                                                      final_position=(3,3),
                                                      blocks=[],
                                                      epsilon=0.1)

    agent = run_sarsa_lambda(alpha=0.01,
                             lamb=0.5,
                             agent=agent,
                             environment=grid,
                             n_episodes=666,
                             gama = 0.1,
                             state_0=(0, 0))

    print('\n')
    print('agent w', agent.return_w())
    agent.epsilon = 0
    state_action_tuple_list, reward_list = run_episode(start_state=(0,0),
                                                       environment=grid,
                                                       agent=agent)

    print('\n number of steps until completion', len(state_action_tuple_list))