class EpisodeSimulator:
    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment


    def run_episode(self, start_state):
        self.environment.set_state(start_state)
        keep_going=True
        state_action_tuple_list = []
        reward_list = []
        while keep_going:
            current_state = self.environment.get_state()
            action = self.agent.return_action(current_state)
            state_action_tuple_list.append((current_state, action))
            self.environment.update_state_with_action(action)
            reward = self.environment.update_state_by_time_and_return_reward()
            print(current_state, action, self.environment.get_state())
            reward_list.append(reward)
            if reward == 0:
                keep_going = False
        return state_action_tuple_list, reward_list








