from exercise_512.track_class import Track
from exercise_512.agent_class import DriverAgent
from exercise_512.episode_simulator import EpisodeSimulator
from numpy import random

grid_height = 10
grid_width = 4

grid = []
for x in range(grid_width):
    for y in range(grid_height):
        grid.append((x, y))

start_locations = [(0,0), (1,0), (2,0), (3,0)]
end_locations = [(3,10), (3,9), (3,8), (0,10), (0,9), (0,10)]

all_possible_states = []
for position in grid:
    for vx in range(5):
        for vy in range(5):
            all_possible_states.append((position, (vx, vy)))


simple_track = Track(start_locations=start_locations,
                     end_locations=end_locations,
                     initial_state=((1, 0),(0,0)),
                     max_speed=4,
                     grid=grid)

possible_actions = [(dvx, dvy) for dvx in [-1, 0 , 1] for dvy in [-1, 0, 1]]
print('possible actions', possible_actions)
upwards_policy = {state:(0,random.choice([0, 1])) for state in all_possible_states}
driver = DriverAgent(initial_policy=upwards_policy,
                     possible_actions=possible_actions,
                     epsilon=0.25)


episode_simulator = EpisodeSimulator(agent = driver, environment=simple_track)

x = episode_simulator.run_episode(start_state=((1, 0),(0,1)))
def test_x():
    state_action = x[0]
    rewards = x[1]
    for i in range(len(state_action)):
        print(state_action[i], '  r:',rewards[i])

