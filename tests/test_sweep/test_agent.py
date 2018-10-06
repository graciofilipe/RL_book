from prioritised_sweep_8.agent import SweepingAgent



# test agent functions
# update_model
# return_model_based_state_action_reward_response
# get_best_action_for_state_from_Q
# return_action
# replace_q_value
# increment_q_value
#  return_state_action_leading_to(self, state_to_query):
def test_update_model(state_action_to_update=('s1', 'a1'),
                      reward_newstate=('r6', 's6')):
    q = {}
    epsilon = 0.1
    possible_actions = [(0,1), (1,0), (1,1), (0,0)]
    model = {('s1', 'a1'): ('r', 's2'),
             ('s2', 'a1'): ('r2', 's3') }
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