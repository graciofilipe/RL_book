"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from car_business import CarBusiness

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

import ray.rllib.agents.ppo as ppo

tf = try_import_tf()



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
    import ray
    import ray.rllib.agents.ppo as ppo
    from ray.tune.logger import pretty_print

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["eager"] = False
    config['env_config'] = {
        'rental_profit': 10,
        'transport_cost':  2,
        'lambda_requests_0':  3,
        'lambda_requests_1':  4,
        'lambda_returns_0':  3,
        'lambda_returns_1':  2,
        'initial_state':  np.array([10, 10])}

    trainer = ppo.PPOTrainer(config=config, env=CarBusiness)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(100):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 30 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    import ipdb; ipdb.set_trace()