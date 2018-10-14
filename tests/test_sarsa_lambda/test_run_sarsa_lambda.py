from sarsa_lambda_127.run_sarsa_lambda import run_sarsa_lambda
from sarsa_lambda_127.generate_agent_and_environment import create_environment_agent_and_states

def test_run_sarsa_lambda_no_crash():
    grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                  initial_position=(0,0),
                                                  final_position=(3,3),
                                                  blocks=[],
                                                  epsilon=0.1)

    run_sarsa_lambda(alpha=0.5,
                     lamb=0.5,
                     agent=agent,
                     environment=grid,
                     n_episodes=3,
                     state_0=(0,0))
