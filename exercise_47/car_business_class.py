import numpy.random as random


class CarBusiness:
    def __init__(self, rental_profit, transport_cost,
                 lambda_requests_1, lambda_requests_2,
                 lambda_returns_1, lambda_returns_2):

        self.rental_profit = rental_profit
        self.transport_cost = transport_cost
        self.lambda_requests_1 = lambda_requests_1
        self.lambda_requests_2 = lambda_requests_2
        self.lambda_returns_1 = lambda_returns_1
        self.lambda_returns_2 = lambda_returns_2
        self.cars_at_1 = 0
        self.cars_at_2 = 0

    def get_cars_returns(self):
        returned_1 = random.poisson(self.lambda_returns_1)
        returned_2 = random.poisson(self.lambda_returns_2)
        return [returned_1, returned_2]

    def get_car_requests(self):
        requested_1 = random.poisson(self.lambda_requests_1)
        requested_2 = random.poisson(self.lambda_requests_2)
        return [requested_1, requested_2]

    def move_cars(self, cars_to_move_1_to_2):
        '''
        :param cars_to_move_1_to_2: integer number of cars to move. If negative, move from 2 to 1
        :return:
        '''
        self.cars_at_1 -= cars_to_move_1_to_2
        self.cars_at_2 += cars_to_move_1_to_2
        