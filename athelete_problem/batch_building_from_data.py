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

states0 = episode_states[0]
actions0 = episode_actions[0]
# import ipdb; ipdb.set_trace()
# save in batches
for episode_idx in range(n_episodes):
    obs = np.array([0, 0, 0])
    prev_action = None
    prev_reward = None
    done = False
    n_actions = len(episode_actions[episode_idx])
    for t, action in enumerate(episode_actions[episode_idx]):
        action = action_to_int_converter[action]
        state_after_action = np.array(episode_states[episode_idx][t+1])
        rew = -1
        if t == n_actions-1:
            done=True
            print('end of episode', episode_idx, 'at time', t, 'and the final state is', state_after_action, 'and the previous state was', obs)
        else:
            done=False
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
            infos={},
            new_obs=state_after_action)
        obs = state_after_action
        prev_action = action
        prev_reward = rew
    writer.write(batch_builder.build_and_reset())
