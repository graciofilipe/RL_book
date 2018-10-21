from sarsa_lambda_127.environment import Environment
from sarsa_lambda_127.agent import Agent


def create_environment_agent_and_states(grid_shape,
                                        initial_position,
                                        final_position,
                                        blocks,
                                        epsilon,
                                        agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)]):

    # create the grid
    grid = Environment(grid_shape=grid_shape,
                       blocks=blocks,
                     initial_position=initial_position,
                     final_position=final_position)


    # create the agent
    agent = Agent(possible_actions=agent_actions,
                  epsilon=epsilon)

    # the sample state is required to get the right shape for w
    agent.initialize_w(sample_state=initial_position, environment=grid)

    return grid, agent

