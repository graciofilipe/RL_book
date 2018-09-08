class PolicyIterator:
    def __init__(self, policy_improver, policy_evaluator, termination_tol):
        self.policy_improver=policy_improver
        self.policy_evaluator = policy_evaluator
        self.termination_tol = termination_tol

    def iterate_policy(self, environment, agent):
        policy_stable = False
        while not policy_stable:
            environment = self.policy_evaluator.run_policy_evaluation(environment=environment,
                                                                      agent=agent,
                                                                    termination_tol=self.termination_tol)
            print('ran policy evaluator')
            print('going for policy improver')
            agent, policy_stable = self.policy_improver.improve_agent_policy(agent=agent,
                                                              environment=environment)
            print('policy_stable:', policy_stable)
            print('ran policy improver')
            if policy_stable:
                print('policy was fine')
            else:
                print('I changed the policy')



        return environment, agent
