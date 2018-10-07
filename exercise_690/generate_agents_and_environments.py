from exercise_690.gridworld_class import Gridworld
from exercise_690.agent_class import Agent


def create_environment_agent_and_states(grid_shape, initial_position, final_position, epsilon,
                                         agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)]):

    # create the grid
    grid = Gridworld(grid_shape=grid_shape,
                     initial_position=initial_position,
                     final_position=final_position)

    all_states = [(x, y) for x in range(grid_shape[0])
                  for y in range(grid_shape[1])]

    state_actions = [(state, action) for state in all_states for action in agent_actions]
    initial_q_dict = {state_action:0 for state_action in state_actions}

    # create the agent
    agent = Agent(possible_actions=agent_actions,
                  initial_q=initial_q_dict,
                  epsilon=epsilon)

    return grid, agent

