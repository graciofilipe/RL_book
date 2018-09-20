from numpy import random
from exercise_512.track_class import Track
from exercise_512.agent_class import DriverAgent
from exercise_512.episode_simulator import EpisodeSimulator
from exercise_512.monte_carlo_offpolicy_estimator import MonteCarloOffPolicyEstimator

def create_envioronment_agent_and_states():
    grid_height = 10
    grid_width = 4

    grid = []
    for x in range(grid_width):
        for y in range(grid_height):
            grid.append((x, y))

    start_locations = [(0, 0), (1, 0), (2, 0), (3, 0)]
    end_locations = [(3, 10), (3, 9), (3, 8), (0, 10), (0, 9), (0, 10)]

    all_possible_states = []
    for position in grid:
        for vx in range(5):
            for vy in range(5):
                print('new state about to be created:', (position, (vx, vy)))
                all_possible_states.append((position, (vx, vy)))

    simple_track = Track(start_locations=start_locations,
                         end_locations=end_locations,
                         initial_state=((1, 0), (0, 0)),
                         max_speed=4,
                         grid=grid)

    possible_actions = [(dvx, dvy) for dvx in [-1, 0, 1] for dvy in [-1, 0, 1]]
    upwards_policy = {state: (0, random.choice([0, 1])) for state in all_possible_states}
    driver = DriverAgent(initial_policy=upwards_policy,
                         possible_actions=possible_actions,
                         epsilon=0.25)

    return simple_track, driver, all_possible_states


def create_random_q_dict_from_env_and_agent(agent, all_possible_states):
    possible_actions = agent.get_list_of_possible_actions()
    q_dict = {}
    for state in all_possible_states:
        for action in possible_actions:
            state_action_pair = (state, action)
        q_dict[state_action_pair]=random.uniform()
    return q_dict

simple_track, driver, all_possible_states = create_envioronment_agent_and_states()

initial_q_dict = create_random_q_dict_from_env_and_agent(agent=driver,
                                                         all_possible_states=all_possible_states)


policy_estimator = MonteCarloOffPolicyEstimator(agent=driver,
                                                environment=simple_track,
                                                q_dict=initial_q_dict)

def test_policy_estimator():

    new_policy = policy_estimator.estimate_policy(gama=0.9,
                                                  max_iter=100)