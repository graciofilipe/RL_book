import gym
import numpy as np
import ray
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn import DQNTrainer

ray.init()

class Coach(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([-5]),
                                                high=np.array([5]),
                                                dtype=np.ndarray)

    def reset(self):
        self.state = np.array([0])
        return self.state

    def setp(self):
        pass

config = DEFAULT_CONFIG
for k, v in config.items():
    print(k, v)

config['hiddens'] = [2]
config['num_workers'] = 1
config['model'] = {'conv_filters': None, 'conv_activation': 'relu', 'fcnet_activation': 'relu', 'fcnet_hiddens': [1],
                   'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': True, 'use_lstm': False,
                   'max_seq_len': 0, 'lstm_cell_size': 0, 'lstm_use_prev_action_reward': False, 'state_shape': None,
                   'framestack': True, 'dim': 0, 'grayscale': False, 'zero_mean': False,
                   'custom_preprocessor': None, 'custom_model': None, 'custom_action_dist': None, 'custom_options': {}}
config['input'] = "/Users/filipe.gracio/projects/RL_book/simple_offline_learn/output/"
config['input_evaluation'] = []
config['exploration_final_eps'] = 0
config['exploration_fraction'] = 0
config['soft_q'] = True
config['softmax_temp'] = 1.0
config['lr'] = 0.1

trainer = DQNTrainer(config=config, env=Coach)
for _ in range(11):
    print(_)
    trainer.train()

def is_state_final(state):
    return state == np.array([5])

def get_next_state_from_action(state, action):
    action_mapper = {0: -1, 1: 1}
    next_state = state + action_mapper[action]
    if next_state < np.array([-5]):
        next_state = np.array([-5])
    return next_state

def simulate_episode(trainer, initial_state):
    list_of_states = [initial_state]
    list_of_actions = []
    state = initial_state
    i = 0
    while (not is_state_final(state) and i < 1000):
        # import ipdb; ipdb.set_trace()
        # print('state', state)
        action = trainer.compute_action(state)
        next_state = get_next_state_from_action(state, action)
        # print('state:', state, '  best_action:', action, '  next_state:', next_state, '    for time', i)
        list_of_actions.append(action)
        list_of_states.append(next_state)
        state = next_state
        i += 1
    # print('learned to reached goal in', i, 'steps')
    return list_of_states, list_of_actions

# import ipdb; ipdb.set_trace()
episode_len_ls = []
for i in range(66):
    episode_states, episode_actions = simulate_episode(trainer, np.array([0]))
    episode_len_ls.append(len(episode_actions))

print('mean len:', np.mean(episode_len_ls))
