
def run_one_step_actor_critic(agent,
                              environment,
                              start_state,
                              n_iter,
                              alpha_th,
                              alpha_w,
                              gama):

    n_params = len(agent.return_w())
    w = [0 for _ in range(n_params)]
    th = [0 for _ in range(n_params)]

    for _ in range(n_iter):
        state = start_state
        environment.set_state(state)
        I = 1
        terminal_flag = False
        while terminal_flag == False:

            action = agent.return_action(state, environment)
            new_state, reward, terminal_flag = environment.return_state_and_reward_post_action(action)
            environment.set_state(new_state)

            new_state_value = agent.state_value_estimate(new_state)
            state_value = agent.state_value_estimate(state)
            if terminal_flag:
                new_state_value = 0
            delta = reward + gama*new_state_value - state_value

            value_estimate_gradient = agent.state_value_estimate_gradient()
            w_increment = alpha_w*delta*value_estimate_gradient
            agent.increment_w(w_increment)

            ln_policy_gradient = agent.ln_policy_gradient(state=state, action=action, environment=environment)
            th_increment = alpha_th*I*delta*ln_policy_gradient
            agent.increment_th(th_increment)

            I = gama*I
            state = new_state

    return agent




