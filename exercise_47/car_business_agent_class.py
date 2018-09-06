class CarBusinessAgent:
    def __init__(self):
        self.policy_dict = {(i, j):0
                       for i in range(0, 33)
                       for j in range(0, 33)}
        self.total_reward = 0


    def return_action(self, state):
        return self.policy_dict[state]

