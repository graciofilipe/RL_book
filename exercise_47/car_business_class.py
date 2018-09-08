import numpy.random as random
from scipy.stats import poisson as poisson
random.seed(111)

class CarBusiness:
    def __init__(self, rental_profit, transport_cost,
                 lambda_requests_0, lambda_requests_1,
                 lambda_returns_0, lambda_returns_1,
                 initial_state):

        self.rental_profit = rental_profit
        self.transport_cost = transport_cost
        self.lambda_requests_0 = lambda_requests_0
        self.lambda_requests_1 = lambda_requests_1
        self.lambda_returns_0 = lambda_returns_0
        self.lambda_returns_1 = lambda_returns_1
        self.state_value_dict = {(i, j): random.uniform(0,10)
                       for i in range(0, 20)
                       for j in range(0, 20)}
        self.state = initial_state

        max_car_requests=5
        self.pmf_requests_0_dict = {i: poisson.pmf(mu=self.lambda_requests_0, k=i) for i in
                               range(-20 * max_car_requests, 20 * max_car_requests)}
        self.pmf_returns_0_dict = {i: poisson.pmf(mu=self.lambda_returns_0, k=i) for i in
                              range(-20 * max_car_requests, 20 * max_car_requests)}
        self.pmf_requests_1_dict = {i: poisson.pmf(mu=self.lambda_requests_1, k=i) for i in
                               range(-20 * max_car_requests, 20 * max_car_requests)}
        self.pmf_returns_1_dict = {i: poisson.pmf(mu=self.lambda_returns_1, k=i) for i in
                              range(-20 * max_car_requests, 20 * max_car_requests)}


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
        '''
        at the end of the day (before action) \
        agent acts , cars are rented, cars are returned, \
        new state is arrived
        :return:
        '''
        # get temp state after action is taken
        state = self.get_state()
        tmp_state_0 = state[0] - action
        tmp_state_1 = state[1] + action

        adjustment_0 = end_state[0] - tmp_state_0
        adjustment_1 = end_state[1] - tmp_state_1

        # loop over the possible ways of getting adjustment_0

        max_car_requests = 12

        probability_list_0 = []
        for i in range(max_car_requests):
            requests = i
            returns = adjustment_0 + i
            p_requests = self.pmf_requests_0_dict[requests]
            p_returns = self.pmf_returns_0_dict[returns]
            probability_list_0.append(p_requests * p_returns)

        probability_list_1 = []
        for i in range(max_car_requests):
            requests = i
            returns = adjustment_1 + i
            p_requests = self.pmf_requests_1_dict[requests]
            p_returns = self.pmf_returns_1_dict[returns]
            probability_list_1.append(p_requests * p_returns)


        final_probabilities = [probability_list_0[i]*probability_list_1[j]
                               for i in range(max_car_requests)
                               for j in range(max_car_requests)]


        final_rewards = [self.rental_profit*(min(i, self.state[0])+min(j,self.state[1]))
                         -self.transport_cost*abs(action)
                         for i in range(max_car_requests) for j in range(max_car_requests)]


        return(final_probabilities, final_rewards)