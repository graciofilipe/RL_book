from exercise_49.gambler_environment import GamblingEnvironment
from exercise_49.gambler_agent import GamblerAgent
from exercise_49.value_iterator_49 import ValueIterator
import pandas as pd


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
                             termination_tol=0.00001)

converged_env, converged_agent = val_iterator.run_value_iteration(environment=gambling_environment,
                                                                  agent=gambling_agent)

print('policy')
print()

state_list = []
action_list = []
print(converged_agent.policy_dict.items())
for state, action in converged_agent.policy_dict.items():
    state_list.append(state)
    action_list.append(action)
df = pd.DataFrame(data={state:state_list,
                        action:action_list})
df.to_csv('exercise_49_actions.csv')

state_list = []
value_list = []
for state, value in converged_env.state_value_dict.items():
    state_list.append(state)
    value_list.append(value)
df = pd.DataFrame(data={state:state_list,
                        value:value_list})
df.to_csv('exercise_49_values.csv')

####
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
pr.print_stats(sort="cumtime")
print(s.getvalue())

