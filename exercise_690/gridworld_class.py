class Gridworld:
    def __init__(self, grid_shape, initial_state, final_state):
        self.grid_shape = grid_shape
        self.state = initial_state
        self.x_bound = grid_shape[0]
        self.y_bound = grid_shape[1]


    def get_state(self):
       return self.state

    def set_state(self, new_state):
        self.state = new_state

    def return_state_and_reward_post_action(self, action):
        new_state = min(self.x_bound, max(0, self.state[0]+action[0])), \
                    min(self.y_bound, max(0, self.state[1]+action[1]))

        reward = -1

        return new_state, reward











