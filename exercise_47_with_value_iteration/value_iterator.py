import numpy as np

class ValueIterator:
    def __init__(self, gama, termination_tol):
        self.gama = gama
        self.termination_tol = termination_tol


    def get_list_of_reacheable_states(self, environment, agent):
        possible_actions = agent.get_list_of_possible_actions()
        list_of_recheable_states = [environment.get_state_after_action(action)
                                    for action in possible_actions]
        return list_of_recheable_states


    def run_value_iteration(self, environment, agent):


        possible_actions = agent.get_list_of_possible_actions()
        list_of_all_states = environment.get_all_possible_states()
        keep_going = True

        while keep_going:
            dif = 0
            for state in list_of_all_states:
                environment.set_state(state)
                old_val = environment.state_value_dict[state]
                action_values = []
                for action in possible_actions:
                    action_value = 0
                    list_of_reacheable_states = self.get_list_of_reacheable_states(environment=environment,
                                                                                   agent=agent)
                    list_of_reacheable_states = list(set(list_of_reacheable_states))
                    for end_state in list_of_reacheable_states:
                        probabilities, rewards = environment.get_probabilities_and_rewards(action=action,
                                                                                           end_state=end_state)
                        pre_value = sum([probabilities[i] * rewards[i] for i in range(len(probabilities))])
                        pre_value += sum(probabilities) * self.gama * environment.get_value_of_state(end_state)
                        action_value += pre_value
                    action_values.append(action_value)
                best_action = possible_actions[np.argmax(action_values)]
                state_value = max(action_values)
                agent.change_policy(state=state,
                                    new_action=best_action)
                environment.update_value_of_state(state=state,
                                                  new_value=state_value)

                pre_diff = abs(old_val  - environment.state_value_dict[state])
                dif = max(dif, pre_diff)

                #print('changed policy for', state, 'from', current_action, 'to', best_action)
                print('current dif', dif)

                if dif < self.termination_tol:
                    keep_going = False

        return environment, agent
