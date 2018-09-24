class Gridworld:
    def __init__(self, grid_shape, initial_state):
        self.grid_shape = grid_shape
        self.state = initial_state


    def get_state(self):
       return self.state

    def update_state(self, new_state):
        self.state = new_state





