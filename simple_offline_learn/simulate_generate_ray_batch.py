import random
import numpy as np

from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
writer = JsonWriter("")
episode_lens_ls = []
action_mapper = {0:-1, 1:1}
width = np.array(5)
n_episodes = 8888
# simulate in batches
for episode_idx in range(n_episodes):
    obs = np.array([0])
    prev_action = None
    prev_reward = None
    done = False
    t = 0
    while not done:
        # print('obs', obs)
        action = random.choice([0, 1])
        # print('actions', action)
        new_obs =  obs + action_mapper[action]
        if new_obs < -5:
            new_obs = np.array([-5])
            # print('capping')
        # print('new_obs', new_obs, 'of type', type(new_obs))
        # print('\n')
        if new_obs == width:
            done=True
            # print('end of episode', episode_idx, 'at time', t)
            episode_lens_ls.append(t)
        else:
            done=False
        batch_builder.add_values(
            t=t,
            eps_id=episode_idx,
            agent_index=0,
            obs=obs,
            actions=action,
            action_prob=1.0,  # put the true action probability here
            rewards=-1*t,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos={},
            new_obs=new_obs)
        obs = new_obs
        prev_action = action
        prev_reward = -1
        t += 1
    writer.write(batch_builder.build_and_reset())

print('average episode lens', np.mean(episode_lens_ls))
