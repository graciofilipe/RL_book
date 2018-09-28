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
            pre_state = self.environment.get_state()
            action = self.agent.return_action(pre_state)
            state_action_tuple_list.append((pre_state, action))
            new_state, reward, terminal_flag = self.environment.return_state_and_reward_post_action(action)
            self.environment.set_state(new_state)
            reward_list.append(reward)
            #print('pre_state', pre_state, 'action', action, 'new_state', new_state)
            if terminal_flag:
                keep_going = False

        return state_action_tuple_list, reward_list
