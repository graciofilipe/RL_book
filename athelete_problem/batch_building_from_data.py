import pickle
import numpy as np

from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
writer = JsonWriter("")

actions = ['strength_1', 'strength_2', 'flexibility_1', 'flexibility_2', 'rest']
action_to_int_converter = {actions[i]: i for i in range(len(actions))}
#
with open('episode_actions.pkl', 'rb') as file:
    episode_actions = pickle.load(file)

with open('episode_states.pkl', 'rb') as file:
    episode_states = pickle.load(file)
n_episodes = len(episode_actions)

# save in batches
for episode_idx in range(n_episodes):
    obs = np.array([0, 0, 0])
    prev_action = 4
    prev_reward = 0
    done = False
    t = 0
    episode_len = len(episode_states[episode_idx])
    while t < episode_len-1:
        # import ipdb;
        # ipdb.set_trace()
        action = action_to_int_converter[episode_actions[episode_idx][t]]
        new_obs = np.array(episode_states[episode_idx][t+1])
        rew = -1
        done = False
        info = {}

        batch_builder.add_values(
            t=t,
            eps_id=episode_idx,
            agent_index=0,
            obs=obs,
            actions=action,
            action_prob=1.0,  # put the true action probability here
            rewards=rew,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            new_obs=new_obs)
        obs = new_obs
        prev_action = action
        prev_reward = rew
        t += 1
    writer.write(batch_builder.build_and_reset())
