'''some useful functions for the sweeping algorithm'''



def add_to_priority_list(current_priority_list, item_to_add):
    rebuilt_list =[]
    # if the list is empty, it's trivial
    if len(current_priority_list)==0:
        rebuilt_list.append(item_to_add)
        return rebuilt_list

    # if the list isn't empty need to check if there is a duplicate
    else:
        append_new_item = True
        state_action_to_add = item_to_add[1]
        priority_of_the_new_item = item_to_add[0]
        for current_item in current_priority_list:
            if current_item[1] == state_action_to_add:
                priority_of_the_current_item = current_item[0]
                if priority_of_the_new_item < priority_of_the_current_item:
                    rebuilt_list.append(current_item)
                    append_new_item = False
            else:
                rebuilt_list.append(current_item)

        # append the new item
        if append_new_item:
            rebuilt_list.append(item_to_add)

        rebuilt_list.sort()
        return(rebuilt_list)




def run_sweep(environment, agent, n_iter, gama, theta, alpha):

    for iteration in range(n_iter):
        p_list = []

        # a
        current_state = environment.get_state()

        # b
        action = agent.return_action(current_state)

        # c
        new_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
        environment.set_state(new_state=new_state)

        # d
        agent.update_model(state_action=(current_state, action),
                           reward_newstate=(reward, new_state))

        # e
        best_action = agent.get_best_action_for_state_from_Q(new_state)
        p = abs(reward + gama*agent.q_dict[(new_state, best_action)] - agent.q_dict[(current_state, action)]
                )
        # f
        if p > theta:
            new_item = (p, (current_state, action))
            p_list = add_to_priority_list(p_list, new_item)

        # g
        while len(p_list) > 0:
            state_action = p_list[-1][1]
            p_list = p_list[:-1]
            reward, new_state = agent.return_model_based_state_action_reward_response(state_action)
            best_action = agent.get_best_action_for_state_from_Q(new_state)
            q_update = alpha * (
                    reward + gama*agent.q_dict[(new_state, best_action)] - agent.q_dict[state_action]
                    )
            agent.increment_q_value(state_action, value_to_increment=q_update)

            states_actions_leading_to_new_state, rewards = agent.return_state_action_leading_to(state_action[0])
            n_actions_leading_to_state = len(states_actions_leading_to_new_state)
            for idx in range(n_actions_leading_to_state):
                preceding_state_action = states_actions_leading_to_new_state[idx]
                reward = rewards[idx]
                best_action = agent.get_best_action_for_state_from_Q(state_action[0])
                p = abs(reward +
                        gama*agent.q_dict[(state_action[0], best_action)] - agent.q_dict[preceding_state_action]
                        )
                if p > theta:
                    new_item = (p, preceding_state_action)
                    p_list = add_to_priority_list(p_list, new_item)
    return agent
