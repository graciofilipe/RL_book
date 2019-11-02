import gym
import numpy as np
import numpy.random as random

random.seed(111)


class Track(gym.Env):
    def __init__(self, end_locations, initial_state, max_speed):

        self.end_locations = end_locations
        self.state = initial_state
        self.max_speed = max_speed
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3)))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=np.array([0, 0]),
                           high=np.array([9, 40]),
                           dtype=np.ndarray),
            gym.spaces.Box(low=np.array([0, 0]),
                           high=np.array([5, 5]),
                           dtype=np.ndarray)
        ))
        self.action_mapping = {0: -1, 1:0, 2:1}

    def step(self, action):

        x_act, y_act = action
        x_acc, y_acc = self.action_mapping[x_act], self.action_mapping[y_act]
        x_pos, y_pos = self.state[0]
        x_vel, y_vel = self.state[1]

        # updates based on action
        x_vel = np.min([self.max_speed, x_vel + x_acc])
        y_vel = np.min([self.max_speed, y_vel + y_acc])

        x_pos += x_vel
        y_pos += y_vel

        if x_pos < 0 or x_pos > 9 or y_pos < 0 or y_pos > 40:
            self.state = (np.array([4, 0]), np.array([0, 0]))
        else:
            self.state = (np.array([x_pos, y_pos]), np.array([x_vel, y_vel]))

        end = False
        reward = -1

        for end_location in self.end_locations:
            if (self.state[0] == end_location).all():
                end = True
                reward = 0
                print('episode over')

        return self.state, reward, end, {}


    def reset(self):
        self.state = self.state = (np.array([4, 0]), np.array([0, 0]))
        return self.state
