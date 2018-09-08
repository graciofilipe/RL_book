class GamblingEnvironment:
    def __init__(self, initial_state_value_dict):
        self.name = 'here is a name'
        self.state_value_dict = initial_state_value_dict

    def get_state(self):
        return (self.state)

    def set_state(self, new_state):
        self.state = new_state

    def get_all_possible_states(self):
        return self.state_value_dict.keys()


    def get_value_of_state(self, state):
        return self.state_value_dict[state]

    def update_value_of_state(self, state, new_value):
        self.state_value_dict[state]=new_value

    def get_probabilities_and_rewards(self, action, end_state):
        current_state = self.get_state()

        if current_state + action == end_state:
