from exercise_49.gambler_environment import GamblingEnvironment
from exercise_49.gambler_agent import GamblerAgent
from exercise_49.value_iterator_49 import ValueIterator


import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

possible_states = range(101)

initial_state_value_dict = {i:0 for i in possible_states}
gambling_environment = GamblingEnvironment(
    initial_state_value_dict=initial_state_value_dict,
    p_win=0.4)

gambling_agent = GamblerAgent(initial_money=10,
                              initial_policy={i:1 for i in possible_states},
                              possible_actions=range(51))

val_iterator = ValueIterator(gama=0.9,
                             termination_tol=0.05)

converged_env, converged_agent = val_iterator.run_value_iteration(environment=gambling_environment,
                                                                  agent=gambling_agent)

print(converged_agent.policy_dict.items())
####
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
pr.print_stats(sort="cumtime")
print(s.getvalue())

