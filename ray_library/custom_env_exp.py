"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym import spaces

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search

tf = try_import_tf()


class Maze44(gym.Env):

    def __init__(self, env_config):
        #print('initing')
        self.end_state = np.array([5, 5])
        self.start_state = np.array([0, 0])
        self.current_state = np.array([0, 0])
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([5, 5]), dtype=np.float32)

    def get_reward(self):
        #print('getting reward')
        if (self.current_state == self.end_state).all():
            return 1
        else:
            return -1

    def get_state(self):
        #print('getting state')
        return self.current_state

    def step(self, action):
        #print('stepping')
        self.take_action(action)
        reward = self.get_reward()
        obs = self.get_state()
        #print('obs', obs)
        episode_over = (self.current_state == self.end_state).all()
        if episode_over:
            self.reset()
        return obs, reward, episode_over, {}

    def take_action(self, action):
        #print('taking action')
        if action == 0:
            # up
            self.current_state[1] = np.min([self.current_state[1] + 1, 5])
        if action == 1:
            # down
            self.current_state[1] = np.max([self.current_state[1] - 1, 0])
        if action == 2:
            # left
            self.current_state[0] = np.max([self.current_state[0] - 1, 0])
        if action == 3:
            # right
            self.current_state[0] = np.min([self.current_state[0] + 1, 5])

    def reset(self):
        self.current_state = np.array([0, 0])
        return np.array([0, 0])



class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tunned = tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "eager": True,
            "env": Maze44,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": grid_search([1e-2, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {}
        },
        checkpoint_at_end=True
    )
    import ipdb
    ipdb.set_trace()
    x=1