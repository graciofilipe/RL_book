import numpy as np

class PolicyImprover:
    def __init__(self, gama):
        self.name = 'this is a name'
        self.gama = gama

    def improve_agent_policy(self, agent, environment):

        policy_stable = True
        possible_actions = agent.get_list_of_possible_actions()
        list_of_states  = environment.get_all_possible_states()
        for state in list_of_states:
            environment.set_state(state)
            current_action = agent.return_action(state)
            action_values = []
            for action in possible_actions:
                action_value = 0
                for end_state in list_of_states:
                    probabilities, rewards = environment.get_probabilities_and_rewards(action=action,
                                                                                       end_state=end_state)
                    pre_value = sum([probabilities[i] * rewards[i] for i in range(len(probabilities))])
                    pre_value += sum(probabilities) * self.gama * environment.get_value_of_state(end_state)
                    action_value += pre_value
                action_values.append(action_value)
            best_action = possible_actions[np.argmax(action_values)]
            if best_action != current_action:
                agent.change_policy(state=state, new_action=best_action)
                policy_stable=False
                print('changed policy for', state, 'from', current_action, 'to', best_action, '\n')
        return agent, policy_stable