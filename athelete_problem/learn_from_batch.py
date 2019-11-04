import gym
import numpy as np
from ray.rllib.agents.pg import DEFAULT_CONFIG
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.offline.wis_estimator import WeightedImportanceSamplingEstimator

class Coach(gym.Env):
    def __init__(self, env_config):

        self.action_space = gym.spaces.Discrete(5)
        self.disc_to_real_converter = {i: i - 5 for i in range(11)}
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0]),
                                                high=np.array([8, 8, 8]),
                                                dtype=np.ndarray)


    def reset(self):
        self.state = np.array([0, 0, 0])
        return self.state

    def setp(self):
        pass

config = DEFAULT_CONFIG
for k, v in config.items():
    print(k, v)
# config['input_evaluation'] = ['wis']
# config['compress_observations'] = False

trainer = PGTrainer(config=config, env=Coach)

estimator = WeightedImportanceSamplingEstimator(trainer.get_policy(), gamma=0.99)
reader = JsonReader("/Users/filipe.gracio/projects/RL_book/athelete_problem/output-2019-11-04_17-35-11_worker-0_0.json")
for _ in range(1000):
    batch = reader.next()
    for episode in batch.split_by_episode():
        print(estimator.estimate(episode))

import ipdb; ipdb.set_trace()
