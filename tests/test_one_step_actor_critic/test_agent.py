from one_step_actor_critic.generate_agent_and_env import create_environment_agent_and_states
import numpy as np
from collections import Counter

grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                  initial_position=(0,0),
                                                  final_position=(3,3),
                                                  blocks=[],
                                                  epsilon=0.1,
                                                  agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])


#a.initialize_w()
def test_initialize_w():
    agent.initialize_w()
    assert len(agent.w) > 0

def test_return_w():
    assert len(agent.return_w())>0

#a.update_w()
def test_update_w():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    n_features = len(agent.return_w())
    new_w = [666 for f in range(n_features)]
    agent.update_w(new_w=new_w)
    assert agent.return_w() == new_w



#a.increment_th()
def test_increment_th():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    n_features = len(agent.return_th())
    increment_th = np.array([111 for _ in range(n_features)])
    agent.increment_th(th_increment=increment_th)
    assert list(agent.return_th())== list(increment_th)


#a.increment_w()
def test_increment_w():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    n_features = len(agent.return_w())
    increment_w = np.array([111 for _ in range(n_features)])
    agent.increment_w(w_increment=increment_w)
    assert list(agent.return_w())== list(increment_w)



def test_state_to_feature_vec():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    state = (0, 0)
    fv = agent.state_to_feature_vec(state=state)
    assert list(fv) == [3, 3, 1]

    state = (0, 3)
    fv = agent.state_to_feature_vec(state=state)
    assert list(fv) == [3, 0, 1]



#a.state_action_to_feature_vec()
def test_state_action_to_feature_vec():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    state_action = ((0, 0), (1, 0))
    fv = agent.state_action_to_feature_vec(state=state_action[0],
                                           action=state_action[1],
                                           environment=grid)
    assert list(fv) == [2, 3, 1]

    state_action = ((2, 0), (-1, 0))
    fv = agent.state_action_to_feature_vec(state=state_action[0],
                                           action=state_action[1],
                                           environment=grid)
    assert list(fv) == [2, 3, 1]

    state_action = ((2, 3), (-1, 1))
    fv = agent.state_action_to_feature_vec(state=state_action[0],
                                           action=state_action[1],
                                           environment=grid)
    assert list(fv) == [2, 0, 1]


#a.state_value_estimate()
def test_state_value_estimate():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    state = (0, 0)
    v = agent.state_value_estimate(state=state)
    # because w is initialized to 0
    assert v == 0

    agent.update_w(new_w=[1,1,1])
    v = agent.state_value_estimate(state=state)
    assert v == 3*1 + 3*1 +1

    state = (1, 1)
    agent.update_w(new_w=[0, 1, 0])
    v = agent.state_value_estimate(state=state)
    assert v == 2*0 + 2*1 + 0*1


#a.from_state_action_to_q_estimate()
def test_from_state_action_to_q_estimate():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    state_action= ((0, 0), (0 ,0))
    v = agent.from_state_action_to_q_estimate(state=state_action[0],
                                              action=state_action[1],
                                              environment=grid)
    # because w is initialized to 0
    assert v == 0

    agent.update_w(new_w=[1,1,1])
    state_action = ((0, 0), (2, 0))
    v = agent.from_state_action_to_q_estimate(state=state_action[0],
                                              action=state_action[1],
                                              environment=grid)
    assert v == 1*3 + 1*1 + 1*1

    agent.update_w(new_w=[1, 10, 1])
    state_action = ((2, 0), (2, 1))
    v = agent.from_state_action_to_q_estimate(state=state_action[0],
                                              action=state_action[1],
                                              environment=grid)
    assert v == 1*0 + 10*2 + 1 * 1


#a.get_state_action_values()
def test_get_state_action_values():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])

    agent.update_th(new_th=[10,10,0])
    sav = agent.get_state_action_values(state_to_interrogate=(0, 0), environment=grid)
    assert sav == [60, 50, 60, 50]

    agent.update_th(new_th=[1, 1, 1])
    sav = agent.get_state_action_values(state_to_interrogate=(3, 0), environment=grid)
    assert sav == [1+3+1, 0+3+1 , 0+3+1 , 0+2+1 ]

    agent.update_th(new_th=[1, 1, 1])
    sav = agent.get_state_action_values(state_to_interrogate=(3, 2), environment=grid)
    assert sav == [1 + 1 + 1, 0 + 1 + 1, 0 + 2 + 1, 1]


#a.get_action_from_policy()
def test_get_action_from_policy():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])

    agent.update_w(new_w=[-10, -10, 1])
    agent.update_th(new_th=[-3, -3, 1])
    a_list = []
    for _ in range(100):
        a_list.append(agent.get_action_from_policy(state_to_interrogate=(0, 0), environment=grid))
    print(Counter(a_list))


def test_state_value_estimate_gradient():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])

    g = agent.state_value_estimate_gradient(state=(0,0))
    assert list(g) == [3, 3, 1]

    g = agent.state_value_estimate_gradient(state=(1, 3))
    assert list(g) == [2, 0, 1]




#a.ln_policy_gradient()
def test_ln_policy_gradient():
    grid, agent = create_environment_agent_and_states(grid_shape=(4, 4),
                                                      initial_position=(0, 0),
                                                      final_position=(3, 3),
                                                      blocks=[],
                                                      epsilon=0.1,
                                                      agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])

    g = agent.ln_policy_gradient(state=(1,1), action=(-1,-1), environment=grid)
    print(g)
    ## TODO: assert the test condition

