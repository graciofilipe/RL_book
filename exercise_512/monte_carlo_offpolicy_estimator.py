from numpy import random
from numpy import argmax
from exercise_512.episode_simulator import EpisodeSimulator

class MonteCarloOffPolicyEstimator:
    def __init__(self, agent, environment, q_dict):
        self.agent = agent
        self.environment = environment
        self.q_dict = q_dict
        self.policy = {}
        self.infer_policy_from_q()
        print(self.policy)

    # infer policy from the q_function
    def infer_policy_from_q(self):
        state_val = {}
        for state_action_tuple, q_val in self.q_dict.items():
            print(q_val)
            state, action = state_action_tuple[0], state_action_tuple[1]
            if q_val > state_val.get(state, -9999):
                self.policy[state] = action
                state_val[state] = q_val
        x=1

    def get_best_action_for_state_from_Q(self, q_dict, state_to_interrogate):
        action_list=[]
        value_list = []
        for state_action, value in q_dict.items():
            state = state_action[0]
            action = state_action[1]
            if state == state_to_interrogate:
                action_list.append(action)
                value_list.append(value)
        arg_max = argmax(value_list)
        return action_list[arg_max]


    def estimate_policy(self, gama, max_iter):
        self.c_dict = {state: 0 for state in self.q_dict.keys()}
        for iter in range(max_iter):
            behavior_policy = self.policy.copy()
            self.agent.set_new_policy(new_policy = behavior_policy)

            g = 0
            w = 1
            # generate episode
            episode_simulator = EpisodeSimulator(agent=self.agent, environment=self.environment)
            random_start_location = self.environment.starting_locations[random.choice(len(self.environment.starting_locations))]
            start_state = (random_start_location, (0, 0))
            state_action_list, rewards_list = episode_simulator.run_episode(start_state=start_state)
            episode_len = len(state_action_list)
            for t in range(episode_len-1, -1, -1):
                g = g*gama + rewards_list[t]
                state = state_action_list[t][0]
                action = state_action_list[t][1]
                self.c_dict[(state, action)] = self.c_dict.get((state, action), 0) + w
                q_update = (w/self.c_dict[(state, action)])*(g - self.q_dict[(state, action)])
                self.q_dict[(state, action)] += q_update
                new_action = self.get_best_action_for_state_from_Q(self.q_dict, state)
                self.agent.change_policy(state=state_action_list[t][0], new_action=new_action)
                self.policy = self.agent.policy_dict
                if new_action != action:
                    break
                else:
                    w=w*(1/1)
        return self.agent.policy_dict









