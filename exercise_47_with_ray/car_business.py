import numpy.random as random
import numpy as np
import gym
random.seed(111)

class CarBusiness(gym.Env):
    def __init__(self, env_config):
        print('yoooooooo')
        self.rental_profit = env_config['rental_profit']
        self.transport_cost = env_config['transport_cost']
        self.lambda_requests_0 = env_config['lambda_requests_0']
        self.lambda_requests_1 = env_config['lambda_requests_1']
        self.lambda_returns_0 = env_config['lambda_returns_0']
        self.lambda_returns_1 = env_config['lambda_returns_1']
        self.state = env_config['initial_state']
        self.action_space = gym.spaces.Discrete(11)
        self.disc_to_real_converter = {i:i-5 for i in range(11)}
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([20, 20]),
                                                dtype=np.ndarray)

    def reset(self):
        self.state = np.array([10, 10])
        return np.array([10, 10])

    def customer_actions(self):
        requests_at_0 = np.random.poisson(lam=self.lambda_requests_0)
        requests_at_1 = np.random.poisson(lam=self.lambda_requests_1)
        rented_at_0 = min([requests_at_0, self.state[0]])
        rented_at_1 = min([requests_at_1, self.state[1]])

        print('state before customer actions', self.state)

        print('requests_at_0', requests_at_0)
        print('requests_at_1', requests_at_1)
        print('rented_at_0', rented_at_0)
        print('rented_at_1', rented_at_1)

        returns_at_0 = np.random.poisson(lam=self.lambda_returns_0)
        returns_at_1 = np.random.poisson(lam=self.lambda_returns_1)

        print('returns_at_0', returns_at_0)
        print('returns_at_1', returns_at_1)
        self.state[0] += returns_at_0 - rented_at_0
        self.state[1] += returns_at_1 - rented_at_1

        print('state after customer actions', self.state)
        # import ipdb; ipdb.set_trace()
        return (rented_at_0 + rented_at_1)*self.rental_profit

    def step(self, action):
        day_profit = self.customer_actions()
        moving_costs = self.take_action(action)
        reward = int(day_profit - moving_costs)
        obs = self.get_state()
        if (obs == np.array([0, 0])).all():
            end = True
            reward = 0
            print('episode over')
        else:
            end = False

        print('obs', obs)
        return obs, reward, end, {}

    def get_state(self):
        return self.state

    def take_action(self, action):
        real_action = self.disc_to_real_converter[action]
        print('taking action', real_action)
        print('state before action', self.state)
        a = np.min([np.max([0, self.state[0] + real_action]), 20])
        b = np.min([np.max([0, self.state[1] - real_action]), 20])
        self.state = np.array([a , b])
        print('state after action', self.state)
        return abs(real_action)*self.transport_cost




