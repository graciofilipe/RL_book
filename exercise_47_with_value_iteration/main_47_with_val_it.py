from exercise_47.car_business_class import CarBusiness
from exercise_47.car_business_agent_class import CarBusinessAgent
from exercise_47_with_value_iteration.value_iterator import ValueIterator


import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

import numpy as np

car_business_environment = CarBusiness(rental_profit=10,
                                       transport_cost=2,
                                       lambda_requests_0=3,
                                       lambda_requests_1=4,
                                       lambda_returns_0=3,
                                       lambda_returns_1=2,
                                       initial_state=(10,10))

car_business_agent = CarBusinessAgent(possible_actions=range(-5, 5))

val_iterator = ValueIterator(gama=0.9,
                             termination_tol=0.5)

converged_env, converged_agent = val_iterator.run_value_iteration(environment=car_business_environment,
                                                                  agent=car_business_agent)

print(converged_agent.policy_dict.items())
####
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
pr.print_stats(sort="cumtime")
print(s.getvalue())

