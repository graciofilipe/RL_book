
def run_episode(start_state, environment, agent):
    environment.set_state(start_state)
    keep_going=True
    state_action_tuple_list = []
    reward_list = []
    while keep_going:
        pre_state = environment.get_state()
        action = agent.return_action(pre_state, environment)
        state_action_tuple_list.append((pre_state, action))
        new_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
        environment.set_state(new_state)
        reward_list.append(reward)
        if terminal_flag:
            keep_going = False

    return state_action_tuple_list, reward_list
