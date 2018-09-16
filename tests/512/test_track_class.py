from exercise_512.track_class import Track
from exercise_512.agent_class import DriverAgent

grid_height = 10
grid_width = 4

grid = []
for x in range(grid_width):
    for y in range(grid_height):
        grid.append((x, y))

start_locations = [(0,0), (1,0), (2,0), (3,0)]
end_locations = [(10,3), (9,3), (8,3)]

all_possible_states = []
for position in grid:
    for vx in range(5):
        for vy in range(5):
            all_possible_states.append((position, (vx, vy)))



def test_no_move_by_time_under_no_speed():
    initial_state = ((1, 0), (0, 0))
    simple_track = Track(start_locations=start_locations,
                         end_locations=end_locations,
                         initial_state=initial_state,
                         grid=grid)
    r = simple_track.update_state_by_time_and_return_reward()
    assert simple_track.get_state() == ((1,0), (0,0))

def test_move_by_time_under_speed():
    initial_state = ((1, 0), (0, 1))
    simple_track = Track(start_locations=start_locations,
                         end_locations=end_locations,
                         initial_state=initial_state,
                         grid=grid)
    r = simple_track.update_state_by_time_and_return_reward()
    assert simple_track.get_state() == ((1,1), (0,1))