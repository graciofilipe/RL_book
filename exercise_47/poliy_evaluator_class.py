import numpy as np

class PolicyEvaluator:
    def __init__(self, gama):
        self.name = 'this is a name'
        self.gama = gama

    def update_state_value(self, environment, state_to_update, agent):
        #get action from agent
        action = agent.return_action(state_to_update)
        # get all states
        list_of_states  = environment.get_all_possible_states()
        new_value = 0
        for end_state in list_of_states:
            probabilities, rewards = environment.get_probabilities_and_rewards(action=action,
                                                                               end_state=end_state)

            pre_value = sum([probabilities[i]*rewards[i] for i in range(len(probabilities))])
            pre_value += sum(probabilities)*self.gama*environment.get_value_of_state(end_state)
            new_value += pre_value

        environment.update_value_of_state(state=state_to_update, new_value=new_value)
        return environment

    def run_policy_evaluation(self, environment, agent, termination_tol):
        list_of_all_states = environment.get_all_possible_states()
        flag = True
        while flag:
            old_val_dict = {state:environment.state_value_dict[state] for state in list_of_all_states}
            for state in list_of_all_states:
                environment = self.update_state_value(environment=environment,
                                                      state_to_update=state,
                                                      agent=agent)
            val_differences = [abs(old_val_dict[state] - environment.state_value_dict[state]) for state in list_of_all_states]
            dif = np.max(val_differences)
            print('dif', dif)
            if dif < termination_tol:
                flag = False
        return environment



