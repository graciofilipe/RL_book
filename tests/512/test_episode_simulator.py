from exercise_512.generate_agents_and_environments import create_envioronment_agent_and_states
from exercise_512.episode_simulator import EpisodeSimulator
from numpy import random


simple_track, driver, states = create_envioronment_agent_and_states()

episode_simulator = EpisodeSimulator(agent = driver, environment=simple_track)

x = episode_simulator.run_episode(start_state=((1, 0),(0,1)))
def test_x():
    state_action = x[0]
    rewards = x[1]
    for i in range(len(state_action)):
        print(state_action[i], '  r:',rewards[i])



