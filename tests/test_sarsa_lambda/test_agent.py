from sarsa_lambda_127.agent import Agent


def test_initialize_w():
    agent_actions = [(0, 1), (0,-1), (1, 0), (-1, 0)]
    epsilon = 0.1
    agent = Agent(possible_actions=agent_actions,
                  epsilon=epsilon)

    sample_state = (0,0)
    n_state_dims = len(sample_state)
    n_action_dims = 2
    agent.initialize_w(sample_state=sample_state)
    n_features = n_state_dims + n_action_dims + n_action_dims*n_state_dims*2
    assert len(agent.w) == n_features


def test_state_action_to_feature_vec():
    agent_actions = [(0, 1), (0,-1), (1, 0), (-1, 0)]
    epsilon = 0.1
    agent = Agent(possible_actions=agent_actions,
                  epsilon=epsilon)

    state = (2, 1)
    action = (0, 1)
    agent.initialize_w(sample_state=state)
    fv = agent.state_action_to_feature_vec(state=state,
                                           action=action)
    feature_vec = [state[0], state[1],
                   action[0], action[1],
                   state[0]*action[0], state[0]*action[1],
                   state[1]*action[0], state[1]*action[1],
                   state[0] - action[0], state[0] - action[1],
                   state[1] - action[0], state[1] - action[1]]

    assert list(fv) == feature_vec


def test_from_state_action_to_q_estimate():
    agent_actions = [(0, 1), (0,-1), (1, 0), (-1, 0)]
    epsilon = 0.1
    agent = Agent(possible_actions=agent_actions,
                  epsilon=epsilon)

    state = (2, 1)
    action = (0, 1)
    agent.initialize_w(sample_state=state)

    new_w = [0 for _ in range(len(agent.w))]
    agent.update_w(new_w=new_w)
    assert agent.from_state_action_to_q_estimate(state=state, action=action) == 0

    new_w = [1 for _ in range(len(agent.w))]
    agent.update_w(new_w=new_w)
    q_est0 = agent.from_state_action_to_q_estimate(state=state, action=action)
    q_est1 = state[0] + state[1] + action[0] + action[1] +\
             state[0] * action[0] + state[0] * action[1]+\
             state[1] * action[0]+ state[1] * action[1]+\
             state[0] - action[0]+ state[0] - action[1]+\
             state[1] - action[0]+ state[1] - action[1]

    assert q_est0==q_est1


def test_get_best_action_for_state_from_Q():
    agent_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    epsilon = 0.1
    agent = Agent(possible_actions=agent_actions,
                  epsilon=epsilon)
    state = (0, 0)
    agent.initialize_w(sample_state=state)
    new_w = [1 for _ in range(len(agent.w))]
    agent.update_w(new_w=new_w)

    best_action_set = {agent.get_best_action_for_state_from_Q(state_to_interrogate=state)}
    print(best_action_set)
    acceptable_answer_set = {(0,-1), (0,-1)}
    assert len(best_action_set.intersection(acceptable_answer_set)) > 0

def test_return_action():
    pass

