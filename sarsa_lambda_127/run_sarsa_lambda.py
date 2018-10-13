#import episode simulator

def run_sarsa_lambda(alpha, lamb, w0,
                     agent, environment,
                     n_episodes, state_0):

    n_features = len(w0)
    w = w0

    for episode in range(n_episodes):

        environment.set_state(state_0)
        action = agent.get_best_action_for_state_from_Q(state_0)
        z = [0 for i in range(w0)]

        while termination_flag == False:

            # take action
            next_state, reward, termination_flag = environment.return_state_and_reward_post_action(action=action)
            environment.set_state(next_state)
            delta = reward

            for feature_indx in range(n_features):
                delta -= w[feature_indx]
                z[feature_indx] =+ 1

            if termination_flag:
                w += alpha*delta*z # TODO: this is scalar vector multiplication
                agent.update_w(w)
            else:
                action = agent.get_best_action_for_state_from_Q(next_state)

                for feature_indx in range(n_features):
                    delta -= gama*w[feature_indx]
                w += alpha*delta*z # TODO: this is scalar vector multiplication
                z += gama*lamb*z # TODO: this is scalar vector multiplication
                state = next_state
                action = next_action













