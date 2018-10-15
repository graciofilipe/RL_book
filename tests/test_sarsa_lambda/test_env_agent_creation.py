from sarsa_lambda_127.generate_agent_and_environment import create_environment_agent_and_states

def test_create_environment_agent_and_states():

    grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                      initial_position=(0, 0),
                                                      blocks = [],
                                                      final_position=(3,3),
                                                      epsilon=0.1)

    # w was initialised
    assert len(agent.w) > 0

