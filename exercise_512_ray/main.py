"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import numpy as np
from track import Track
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
                       'fcnet_hiddens': [12, 12, 12], 'free_log_std': False, 'no_final_linear': False,
                       'vf_share_layers': True, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256,
                       'lstm_use_prev_action_reward': False, 'state_shape': None, 'framestack': True, 'dim': 4,
                       'grayscale': False, 'zero_mean': True, 'custom_preprocessor': None, 'custom_model': None,
                       'custom_action_dist': None, 'custom_options': {}}
    config['gamma'] = 0.9
    config['num_workers'] = 6
    config['env_config'] = {
        'end_locations': [
            np.array([9, 40]), np.array([9, 39]), np.array([9, 38]), np.array([9, 37])],
        'initial_state': (np.array([4, 0]), np.array([0, 0])),
        'max_speed': 5
    }

    trainer = ppo.PPOTrainer(config=config, env=Track)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(222):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        # if i % 50 == 0:
        #     checkpoint = trainer.save()
        #     print("checkpoint saved at", checkpoint)
    # zerozero = [trainer.compute_action(np.array([10, 10])) for _ in range(10000)]
    # ntzero = [trainer.compute_action(np.array([19, 0])) for _ in range(10000)]
    # zeront = [trainer.compute_action(np.array([0, 19])) for _ in range(10000)]
    # from collections import Counter
    # print(Counter(zerozero))
    # print(Counter(ntzero))
    # print(Counter(zeront))

    tr = Track(config['env_config'])
    tr.play_episode(trainer)