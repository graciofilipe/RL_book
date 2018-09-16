from numpy import random

class Track:
    def __init__(self, start_locations, end_locations, initial_state, grid, max_speed):
        '''
        :param starting_states: [(x,y),(vx,vy)]
        :param end_locations:
        :param initial_state:
        '''
        random.choice(3)
        self.starting_locations = start_locations
        self.end_states = end_locations
        self.state = initial_state
        self.grid = grid
        self.max_speed = max_speed

    def get_state(self):
        return self.state

    def get_location(self):
        return self.state[0]

    def get_speed(self):
        return self.state[1]

    def set_state(self, state):
        self.state = state

    def set_location(self, location):
        self.state = (location, self.get_speed())

    def set_speed(self, speed):
        self.state = (self.get_location(), speed)

    def update_state_with_action(self, action):
        speed = self.get_speed()
        new_speed_0, new_speed_1 = max(0, min(speed[0] + action[0], self.max_speed)),\
                                   max(0, min(speed[1] + action[1],self.max_speed))
        self.set_speed((new_speed_0, new_speed_1))

    def update_state_by_time_and_return_reward(self):
        speed = self.get_speed()
        location = self.get_location()
        new_location = (location[0]+speed[0], location[1]+speed[1])

        if new_location in self.end_states:
            return 0
        else:
            if new_location not in self.grid:
                new_location = self.starting_locations[random.choice(len(self.starting_locations))]
                self.set_speed((0, 0))
            self.set_location(new_location)
            return -1













