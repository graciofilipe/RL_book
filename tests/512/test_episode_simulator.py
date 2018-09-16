from exercise_512.track_class import Track
from exercise_512.agent_class import DriverAgent
from exercise_512.episode_simulator import EpisodeSimulator


grid_height = 10
grid_width = 4

grid = []
for x in range(grid_width):
    for y in range(grid_height):
        grid.append((x, y))

start_locations = [(0,0), (1,0), (2,0), (3,0)]
end_locations = [(10,3), (9,3), (8,3)]

simple_track = Track(start_locations=start_locations,
                     end_locations=end_locations,
                     initial_state=(1, 0),
                     grid=grid)




