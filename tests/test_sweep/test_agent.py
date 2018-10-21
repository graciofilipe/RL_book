from prioritised_sweep_8.agent import SweepingAgent

def test_update_model(state_action_to_update=('s1', 'a1'),
                      reward_newstate=('r6', 's6')):
    q = {}
    epsilon = 0.1
    possible_actions = [(0,1), (1,0), (1,1), (0,0)]
    model = {('s1', 'a1'): ('r', 's2'),
             ('s2', 'a1'): ('r2', 's3')}

    sa = SweepingAgent(q_initial=q, epsilon=epsilon,
                       possible_actions=possible_actions, model=model)

    sa.update_model(state_action=state_action_to_update,
                    reward_newstate=reward_newstate)

    assert sa.model[state_action_to_update] == reward_newstate


def test_return_model_based_state_action_reward_response():
    state_action_for_model = ('s1', 'a1')
    model_prediction = ('r', 's2')

    q = {}
    epsilon = 0.1
    possible_actions = [(0, 1), (1, 0), (1, 1), (0, 0)]
    model = {state_action_for_model: model_prediction,
             ('s2', 'a1'): ('r2', 's3')}
    sa = SweepingAgent(q_initial=q, epsilon=epsilon,
                       possible_actions=possible_actions, model=model)

    assert model_prediction == sa.return_model_based_state_action_reward_response(state_action=state_action_for_model)


def test_get_best_action_for_state_from_Q():
    q = {((0, 0), (0, 1)): 1,
         ((0, 0), (3, 0)): 1.0,
         ((0, 3), (0, -1)): -9,
         ((0, 3), (0, 1)): -8,
         ((0, 0), (-1, 0)): 1,
         ((0, 0), (0, -2)): -9,
         ((0, 0), (1, 0)): 1.2}

    epsilon = 0.1
    possible_actions = [(0, 1), (1, 0), (1, 1), (0, 0)]
    model = {}
    sa = SweepingAgent(q_initial=q,
                       epsilon=epsilon,
                       possible_actions=possible_actions,
                       model=model)

    assert sa.get_best_action_for_state_from_Q(state_to_interrogate=(0, 0)) == (1, 0)
    assert sa.get_best_action_for_state_from_Q(state_to_interrogate=(0, 3)) == (0, 1)

def test_return_action():
    q = {((0, 0), (0, 1)): 1,
         ((0, 0), (3, 0)): 1.0,
         ((0, 3), (0, -1)): -9,
         ((0, 3), (0, 1)): -8,
         ((0, 0), (-1, 0)): 1,
         ((0, 0), (0, -2)): -9,
         ((0, 0), (1, 0)): 1.2}

    epsilon = 0
    possible_actions = [(0, 1), (1, 0), (1, 1), (0, 0)]
    model = {}
    sa = SweepingAgent(q_initial=q,
                       epsilon=epsilon,
                       possible_actions=possible_actions,
                       model=model)

    assert sa.return_action(state=(0,3)) == (0, 1)
    assert sa.return_action(state=(0, 0)) == (1, 0)

    sa.epsilon = 1
    s = set()
    for i in range(1000):
        a = sa.return_action(state=(0,3))
        s.add(a)

    assert len(s)==len(possible_actions)


def test_replace_q_value():
    q = {((0, 0), (0, 1)): 1,
         ((0, 0), (3, 0)): 1.0,
         ((0, 3), (0, -1)): -9,
         ((0, 3), (0, 1)): -8,
         ((0, 0), (-1, 0)): 1,
         ((0, 0), (0, -2)): -9,
         ((0, 0), (1, 0)): 1.2}

    epsilon = 0
    possible_actions = [(0, 1), (1, 0), (1, 1), (0, 0)]
    model = {}
    sa = SweepingAgent(q_initial=q,
                       epsilon=epsilon,
                       possible_actions=possible_actions,
                       model=model)

    sa.replace_q_value(state_action_to_update=((0, 0), (1, 0)),
                       value=666)

    assert sa.q_dict[((0, 0), (1, 0))] == 666


def test_increment_q_value():
    q = {((0, 0), (0, 1)): 1,
         ((0, 0), (3, 0)): 1.0,
         ((0, 3), (0, -1)): -9,
         ((0, 3), (0, 1)): -8,
         ((0, 0), (-1, 0)): 1,
         ((0, 0), (0, -2)): -9,
         ((0, 0), (1, 0)): 1.2}

    epsilon = 0
    possible_actions = [(0, 1), (1, 0), (1, 1), (0, 0)]
    model = {}
    sa = SweepingAgent(q_initial=q,
                       epsilon=epsilon,
                       possible_actions=possible_actions,
                       model=model)

    sa.increment_q_value(state_action_to_update=((0, 3), (0, -1)), value_to_increment=9)

    assert sa.q_dict[((0, 3), (0, -1))] == 0


def test_return_state_action_leading_to():
    q = {((0, 0), (0, 1)): 1,
         ((0, 0), (0, -1)): 1.0,
         ((0, 0), (1, 0)): -9,
         ((0, 0), (-1, 0)): -8,
         ((0, 1), (0, 1)): 1,
         ((0, -1), (0, -1)): 1.0,
         ((1, 0), (1, 0)): -9,
         ((-1, 0), (-1, 0)): -8
         }


    epsilon = 0
    possible_actions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    model = {((0, 0), (0, 1)): (1, (0, 1)),
         ((0, 0), (0, -1)): (1.0, (0, -1)),
         ((0, 0), (1, 0)): (-9, (1, 0)),
         ((0, 0), (-1, 0)): (-8,(-1, 0)),
         ((0, 1), (0, 1)): (1,(0, 2)),
         ((0, -1), (0, -1)): (1.0,(0, -2)),
         ((1, 0), (1, 0)): (-9,(2, 0)),
         ((-1, 0), (-1, 0)): (-8,(-2, 0))
         }
    sa = SweepingAgent(q_initial=q,
                       epsilon=epsilon,
                       possible_actions=possible_actions,
                       model=model)

    preceding_state_actions, rewards = sa.return_state_action_leading_to((0, -1))
    assert preceding_state_actions == [((0, 0), (0, -1))]
    assert rewards == [1.0]