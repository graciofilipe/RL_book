from exercise_512.track_class import Track

track_class = Track(starting_states = [(0,0), (0,1)],
                    end_locations=[(2, 2), (3, 2)],
                    initial_state=(0,0))

def test_that_initial_state_sets_state():
    assert track_class.get_state() == (0, 0)
