class Sarsa:
    def __init__(self, alpha, epsilon):
        self.alpha = alpha
        self.epsilon = epsilon

    def run_sarsa(self, environment, agent, initial_state, n_episodes):

        for episode in n_episodes:
            keep_going = True
            environment.set_state(initial_state)
            action = agent.policy[initial_state]

            while keep_going:
                current_state = environment.get_state()
                new_state, reward = environment.return_state_and_reward_post_action(action)
                next_action = agent.policy[new_state]
                q_update = self.alpha*(reward +\
                                       self.gama*agent.q_dict[(new_state, next_action)] -
                                       self.gama*agent.q_dict[(current_state, action)])
                agent.increment_q_value((current_state, action), q_update)
                environment.set_state(new_state)
                action = next_action
                if new_state == environment.final_position:
                    keep_going = False

        return agent, environment
