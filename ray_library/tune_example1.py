import ray
from ray import tune

ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 6,
        "lr": tune.grid_search([0.001, 0.0001]),
        "eager": True,
    },
)