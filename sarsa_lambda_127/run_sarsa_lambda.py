import numpy as np

def run_sarsa_lambda(alpha, lamb, gama,
                     agent, environment,
                     n_episodes, state_0):

    w = agent.return_w()
    n_features = len(w)


    for episode in range(n_episodes):

        environment.set_state(state_0)
        action = agent.return_action(state_0, environment=environment)
        feature_vec = agent.state_action_to_feature_vec(state=state_0,
                                                        action=action,
                                                        environment=environment)
        z = np.array([0 for _ in range(n_features)])
        q_old = 0

        termination_flag = False
        while termination_flag == False:

            # take action
            next_state, reward, termination_flag = environment.return_state_and_reward_post_action(action=action)
            environment.set_state(next_state)

            next_action = agent.return_action(next_state, environment=environment)

            next_feature_vec = agent.state_action_to_feature_vec(state=next_state,
                                                                 action=next_action,
                                                                 environment=environment)

            if termination_flag:
                next_feature_vec = np.array([0 for _ in range(n_features)])

            q = np.dot(agent.return_w(), feature_vec)
            q_next = np.dot(agent.return_w(), next_feature_vec)

            delta = reward + gama*q_next - q

            fv_scaling = (1 - alpha * gama * lamb * np.dot(z, feature_vec))
            z = gama * lamb * z + fv_scaling * feature_vec
            aux = z

            z_w = alpha * (delta + q - q_old) * z
            fv_w = alpha * (q_old - q) * feature_vec
            w_inc = z_w + fv_w
            w = agent.return_w() + w_inc
            aux2 = w

            agent.update_w(w)
            q_old = q_next
            feature_vec = next_feature_vec
            action = next_action

    return agent