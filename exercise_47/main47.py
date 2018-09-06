#Jack manages two locations for a nationwide car rental company.
# Each day, some number of customers arrive at each location to rent cars.
#If Jack has a car available, he rents it out and is credited $10 by the national company.
#If he is out of cars at that location, then the business is lost. Cars become available for
#renting the day after they are returned. To help ensure that cars are available where
#they are needed, Jack can move them between the two locations overnight, at a cost of
#$2 per car moved. We assume that the number of cars requested and returned at each
#location are Poisson random variables, meaning that the probability that the number is
#n #n is n! e , where is the expected number. Suppose is 3 and 4 for rental requests at
#the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
#we assume that there can be no more than 20 cars at each location (any additional cars
#are returned to the nationwide company, and thus disappear from the problem) and a
#maximum of five cars can be moved from one location to the other in one night


#Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s car
#rental problem with the following changes. One of Jack’s employees at the first location
#rides a bus home each night and lives near the second location. She is happy to shuttle
#one car to the second location for free. Each additional car still costs $2, as do all cars
#moved in the other direction. In addition, Jack has limited parking space at each location.
#If more than 10 cars are kept overnight at a location (after any moving of cars), then an
#additional cost of $4 must be incurred to use a second parking lot (independent of how
#many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often
#occur in real problems and cannot easily be handled by optimization methods other than
#dynamic programming. To check your program, first replicate the results given for the
#original problem.

from exercise_47.car_business_class import CarBusiness
from exercise_47.car_business_agent_class import CarBusinessAgent
from exercise_47.poliy_evaluator_class import PolicyEvaluator
from exercise_47.poliy_improver_class import PolicyImprover



import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

import numpy as np

cb = CarBusiness(rental_profit=10,
                 transport_cost=2,
                 lambda_requests_0=3,
                 lambda_requests_1=4,
                 lambda_returns_0=3,
                 lambda_returns_1=2,
                 initial_state=(10,10))

cba = CarBusinessAgent()
pol_eval = PolicyEvaluator(gama=0.9)
#prob=0
#for i in range(30):
#    for j in range(30):
#        x = cb.get_probabilities_and_rewards(action=0,
#                                             end_state=(i,j))
#        prob += np.sum(x[0])
#print(prob)

updated_env = pol_eval.run_policy_evaluation(environment=cb,
                                             agent=cba,
                                             termination_tol=1)

####
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
pr.print_stats(sort="cumtime")
print(s.getvalue())

print(updated_env.state_value_dict.items())