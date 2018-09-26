class Sarsa:
    def __init__(self, alpha, gama):
        self.gama = gama
        self.alpha = alpha

    def run_sarsa(self, environment, agent, initial_state, n_episodes):

        for episode in range(n_episodes):
            keep_going = True
            environment.set_state(initial_state)
            action = agent.return_action(initial_state)

            while keep_going:
                current_state = environment.get_state()
                new_state, reward, termination_flag = environment.return_state_and_reward_post_action(action)
                next_action = agent.return_action(new_state)
                q_update = self.alpha*(reward +\
                                       self.gama*agent.q_dict[(new_state, next_action)] -
                                       self.gama*agent.q_dict[(current_state, action)])
                agent.increment_q_value((current_state, action), q_update)
                environment.set_state(new_state)
                action = next_action
                if termination_flag:
                    keep_going = False

        return environment, agent
