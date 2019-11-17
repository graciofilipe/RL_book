import gym
import numpy as np
from data_generation import get_next_state_from_action
from ray.rllib.agents.dqn import DEFAULT_CONFIG
from ray.rllib.agents.dqn import DQNTrainer
from collections import Counter
import ray
ray.init()
actions = ['strength_1', 'strength_2', 'flexibility_1', 'flexibility_2', 'rest']
int_to_action_converter = {i: actions[i] for i in range(len(actions))}
np.random.seed(123)
class Coach(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]),
                                                high=np.array([7, 7, 7]),
                                                dtype=np.ndarray)
    def reset(self):
        self.state = np.array([0, 0, 0])
        return self.state
    def setp(self):
        pass


def is_state_final(state):
    return state[0] >= 6 and state[1] >= 6

def simulate_episode(trainer, initial_state):
    list_of_states = [initial_state]
    list_of_actions = []
    state = initial_state
    i = 0
    while (not is_state_final(state) and i < 1000):
        # print('state', state, 'of type', type(state))
        action = trainer.compute_action(state)
        text_based_action = int_to_action_converter[action]
        next_state = np.array(get_next_state_from_action(state, text_based_action))
        # print('state:', state, '  best_action:', text_based_action, '  next_state:', next_state, '    for action', i)
        list_of_actions.append(text_based_action)
        list_of_states.append(next_state)
        state = next_state
        i += 1
    print('learned to reached goal in', i, 'steps')
    return list_of_states, list_of_actions

config = DEFAULT_CONFIG
for k, v in config.items():
    print(k, v)

# config['batch_mode'] = 'complete_episodes'
config['hiddens'] = []
# config['n_step'] = 3
config['num_workers'] = 4
config['input'] = "/Users/filipe.gracio/projects/RL_book/athelete_problem/output/"
config['input_evaluation'] = []
config["evaluation_config"] = {
    'exploration_final_eps': 0,
    'exploration_fraction': 0.1}
# config["schedule_max_timesteps"] = 10000,
# config["timesteps_per_iteration"] = 100,
config['soft_q'] = True
# config['softmax_temp'] = 1
# config['lr'] = 0.005
config['learning_starts'] = 10
config['train_batch_size'] = 6666
# config['output'] = 'logdir'
config['env'] = Coach
config["dueling"] = True
config["double_q"] = True
config["noisy"] = False
config["sigma0"] = 0.5
config['model'] = {'conv_filters': None,
                   'conv_activation': 'relu',
                   'fcnet_activation': 'tanh',
                   'fcnet_hiddens': [3,3],
                   'free_log_std': False,
                   'no_final_linear': False,
                   'vf_share_layers': True,
                   'use_lstm': False,
                   'max_seq_len': 0,
                   'lstm_cell_size': 0,
                   'lstm_use_prev_action_reward': False,
                   'state_shape': None,
                   'framestack': False,
                   'dim': 0,
                   'grayscale': False,
                   'zero_mean': False,
                   'custom_preprocessor': None,
                   'custom_model': None,
                   'custom_action_dist': None,
                   'custom_options': {}}

# episode_len_ls = []
# for i in range(11):
#     trainer = DQNTrainer(config=config, env=Coach)
#     episode_states, episode_actions = simulate_episode(trainer, np.array([0, 0 ,0]))
#     episode_len_ls.append(len(episode_actions))
#     print(Counter(episode_actions))
# print('mean len:', np.mean(episode_len_ls))

#### training #####
trainer = DQNTrainer(config=config, env=Coach)
for i in range(88):
    print('train iteration', i)
    trainer.train()
###################

episode_len_ls = []
for i in range(33):
    episode_states, episode_actions = simulate_episode(trainer, np.array([0, 0 ,0]))
    episode_len_ls.append(len(episode_actions))
    print(Counter(episode_actions))
print('mean len:', np.mean(episode_len_ls))
