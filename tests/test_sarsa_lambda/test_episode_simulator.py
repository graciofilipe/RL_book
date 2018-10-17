from sarsa_lambda_127.sarsa_lambda_episode_simulator import run_episode
from sarsa_lambda_127.generate_agent_and_environment import create_environment_agent_and_states


def test_episode_simulator():
    grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                      initial_position=(0,0),
                                                      final_position=(3,3),
                                                      blocks=[],
                                                      epsilon=0.1)

    state_action_tuple_list, reward_list = run_episode(start_state=(0,0), environment=grid, agent=agent)
    for sa in state_action_tuple_list:
        print(sa)