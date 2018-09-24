class Sarsa:
    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

    def run_sarsa(self, environment, agent, initial_state, n_episodes):

        for episode in n_episodes:
            keep_going = True
            action = agent.policy[initial_state]
            while keep_going:
                current_state = environment.get_state()
                new_state, reward = environment.return_state_and_reward_post_action(action)
                next_action = agent.policy[new_state]

                # update Q
                q_update = self.alpha*(reward + \
                                       self.gama*agent.q_dict[(new_state, next_action)] -
                                       self.gama*agent.q_dict[(current_state, action)])

                agent.q_dict[(current_state, action)] += q_update

                #move state
                environment.set_state(new_state)
                #update action
                action = next_action





