"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
from car_business import CarBusiness
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

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
    config['model'] = {'conv_filters': None, 'conv_activation': 'relu', 'fcnet_activation': 'relu',
                       'fcnet_hiddens': [4, 4, 4], 'free_log_std': False, 'no_final_linear': False,
                       'vf_share_layers': True, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                       'lstm_use_prev_action_reward': False, 'state_shape': None, 'framestack': True, 'dim': 4,
                       'grayscale': False, 'zero_mean': True, 'custom_preprocessor': None, 'custom_model': None,
                       'custom_action_dist': None, 'custom_options': {}}
    config['gamma'] = 0.9
    config['num_workers'] = 1
    config['env_config'] = {
        'rental_profit': 10,
        'transport_cost': 2,
        'lambda_requests_0': 3,
        'lambda_requests_1': 4,
        'lambda_returns_0': 3,
        'lambda_returns_1': 2,
        'initial_state': np.array([10, 10])
    }

    trainer = ppo.PPOTrainer(config=config, env=CarBusiness)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(200):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        # if i % 50 == 0:
        #     checkpoint = trainer.save()
        #     print("checkpoint saved at", checkpoint)
    zerozero = [trainer.compute_action(np.array([10, 10])) for _ in range(10000)]
    ntzero = [trainer.compute_action(np.array([19, 0])) for _ in range(10000)]
    zeront = [trainer.compute_action(np.array([0, 19])) for _ in range(10000)]
    from collections import Counter
    print(Counter(zerozero))
    print(Counter(ntzero))
    print(Counter(zeront))
    import ipdb; ipdb.set_trace()
