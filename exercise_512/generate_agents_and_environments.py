from exercise_512.track_class import Track
from exercise_512.agent_class import DriverAgent
from numpy import random


def create_envioronment_agent_and_states():
    grid_height = 3
    grid_width = 3

    grid = []
    for x in range(grid_width):
        for y in range(grid_height):
            grid.append((x, y))

    start_locations = [(1, 0)]
    end_locations = [(2, 2)]

    all_possible_states = []
    for position in grid:
        for vx in range(3):
            for vy in range(3):
                all_possible_states.append((position, (vx, vy)))

    simple_track = Track(start_locations=start_locations,
                         end_locations=end_locations,
                         initial_state=((1, 0), (0, 0)),
                         max_speed=2,
                         grid=grid)

    possible_actions = [(dvx, dvy) for dvx in [-1, 0, 1] for dvy in [-1, 0, 1]]
    upwards_policy = {state: (random.choice([-1, 0, 1]), random.choice([0, 1])) for state in all_possible_states}
    driver = DriverAgent(initial_policy=upwards_policy,
                         possible_actions=possible_actions,
                         epsilon=0.1)

    return simple_track, driver, all_possible_states