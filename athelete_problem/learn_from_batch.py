import gym
import numpy as np
from data_generation import get_next_state_from_action
from ray.rllib.agents.pg import DEFAULT_CONFIG
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator

actions = ['strength_1', 'strength_2', 'flexibility_1', 'flexibility_2', 'rest']
int_to_action_converter = {i: actions[i] for i in range(len(actions))}


class Coach(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(5)
        self.disc_to_real_converter = {i: i - 5 for i in range(11)}
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]),
                                                high=np.array([18, 18, 18]),
                                                dtype=np.ndarray)

    def reset(self):
        self.state = np.array([0, 0, 0])
        return self.state

    def setp(self):
        pass


config = DEFAULT_CONFIG
for k, v in config.items():
    print(k, v)
config['model'] = {'conv_filters': None, 'conv_activation': 'relu', 'fcnet_activation': 'tanh', 'fcnet_hiddens': [4, 4],
 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True, 'use_lstm': False, 'max_seq_len': 20,
 'lstm_cell_size': 256, 'lstm_use_prev_action_reward': False, 'state_shape': None, 'framestack': True, 'dim': 84,
 'grayscale': False, 'zero_mean': True, 'custom_preprocessor': None, 'custom_model': None, 'custom_action_dist': None,
 'custom_options': {}}

trainer = PGTrainer(config=config, env=Coach)

estimator = WeightedImportanceSamplingEstimator(trainer.get_policy(), gamma=0.99)
reader = JsonReader("/Users/filipe.gracio/projects/RL_book/athelete_problem/output-2019-11-04_19-09-33_worker-0_0.json")
for _ in range(1000):
    batch = reader.next()
    for episode in batch.split_by_episode():
        print(estimator.estimate(episode))


def is_state_final(state):
    return state[0] >= 6 and state[1] >= 6


def simulate_episode(trainer, initial_state):
    list_of_states = [initial_state]
    list_of_actions = []
    state = initial_state
    i = 0
    while (not is_state_final(state) and i < 1000):
        action = trainer.compute_action(state)
        text_based_action = int_to_action_converter[action]
        next_state = get_next_state_from_action(state, text_based_action)
        print('state:', state, '   best_action:', text_based_action, '     next_state:', next_state, '    for action',
              i)
        list_of_actions.append(text_based_action)
        list_of_states.append(next_state)
        state = tuple(next_state)
        i += 1
    print('learned to reached goal in', i, 'steps')
    return list_of_states, list_of_actions


episode_states, episode_actions = simulate_episode(trainer, np.array([0, 0, 0]))
