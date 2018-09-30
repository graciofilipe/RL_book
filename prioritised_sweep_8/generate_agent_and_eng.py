from prioritised_sweep_8.agent import SweepingAgent
from prioritised_sweep_8.env_class import Environment


def create_environment_agent_and_states(grid_shape, initial_position, final_position, epsilon,
                                         agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)]):

    # create the grid
    grid = Environment(grid_shape=grid_shape,
                       initial_position=initial_position,
                       final_position=final_position,
                       blocks = [(2,2), (3, 3), (3, 4), (4, 3)])

    all_states = [(x, y) for x in range(grid_shape[0])
                  for y in range(grid_shape[1])]

    state_actions = [(state, action) for state in all_states for action in agent_actions]
    initial_q_dict = {state_action:0 for state_action in state_actions}
    initial_model = {state_action:(0, (0, 0)) for state_action in state_actions}

    # create the agent
    agent = SweepingAgent(possible_actions=agent_actions,
                          q_initial=initial_q_dict,
                          epsilon=epsilon,
                          model = initial_model)

    return grid, agent

