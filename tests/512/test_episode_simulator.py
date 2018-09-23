from exercise_512.generate_agents_and_environments import create_envioronment_agent_and_states
from exercise_512.episode_simulator import EpisodeSimulator



def test_penultimate_state_plus_action_equals_final_location(n=100):
    for trial in range(n):
        simple_track, driver, states = create_envioronment_agent_and_states()
        max_speed = simple_track.max_speed
        episode_simulator = EpisodeSimulator(agent=driver, environment=simple_track)
        x = episode_simulator.run_episode(start_state=((1, 0), (0, 0)))
        state_action = x[0]
        penultimate_state, last_action =  state_action[-1]
        penultimate_location, penultimate_speed = penultimate_state[0], penultimate_state[1]
        last_speed = (min(max(0, penultimate_speed[0]+last_action[0]), max_speed),
                      min(max(0, penultimate_speed[1]+last_action[1]), max_speed))
        ending_spot = (penultimate_location[0]+last_speed[0], penultimate_location[1]+last_speed[1])

        assert ending_spot == simple_track.end_locations[0]

#def print_an_episode():
#    simple_track, driver, states = create_envioronment_agent_and_states()
#    episode_simulator = EpisodeSimulator(agent=driver, environment=simple_track)
#    x = episode_simulator.run_episode(start_state=((1, 0), (0, 0)))
#    state_action = x[0]
#    rewards = x[1]
#    for i in range(len(state_action)):
#        print(state_action[i], '  r:',rewards[i])



