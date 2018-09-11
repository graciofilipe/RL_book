class GamblingEnvironment:
    def __init__(self, initial_state_value_dict, p_win):
        self.name = 'here is a name'
        self.state_value_dict = initial_state_value_dict
        self.p_win = p_win

    def get_state(self):
        return (self.state)

    def set_state(self, new_state):
        self.state = new_state

    def get_all_possible_states(self):
        return self.state_value_dict.keys()

    def get_currently_permissable_actions(self):
        limit = min(self.state, 100-self.state)
        return range(limit+1)

    def get_currently_recheable_states(self):
        currently_permissable_actions = self.get_currently_permissable_actions()
        currently_recheable_states = []
        for action in currently_permissable_actions:
            currently_recheable_states.append(self.state + action)
            currently_recheable_states.append(self.state - action)
        return currently_recheable_states

    def get_value_of_state(self, state):
        return self.state_value_dict[state]

    def update_value_of_state(self, state, new_value):
        self.state_value_dict[state]=new_value

    def get_probabilities_and_rewards(self, action, end_state):
        current_state = self.get_state()

        ## when end state is reacheable from wining the bet
        if current_state + action == end_state:
            if end_state == 100:
                reward = [1]
            else:
                reward = [0]
            return [self.p_win], reward

        ## when end state is reacheable from losing the bet
        elif current_state - action == end_state:
            return [1-self.p_win], [0]

        else:
            return [0], [0]
