class Gridworld:
    def __init__(self, grid_shape, initial_position, final_position):
        self.grid_shape = grid_shape
        self.initial_position = initial_position
        self.final_position = final_position
        self.state = initial_position
        self.x_bound = grid_shape[0] -1
        self.y_bound = grid_shape[1] -1


    def get_state(self):
       return self.state

    def set_state(self, new_state):
        self.state = new_state

    def return_state_and_reward_post_action(self, action):

        new_state = min(self.x_bound, max(0, self.state[0]+action[0])), \
                    min(self.y_bound, max(0, self.state[1]+action[1]))

        if new_state == self.final_position:
            reward = 0
            terminal_flag = True
        else:
            reward = -1
            terminal_flag = False

        return new_state, reward, terminal_flag











