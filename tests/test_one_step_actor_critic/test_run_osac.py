from one_step_actor_critic.run_one_step_actor_critic import run_one_step_actor_critic
from one_step_actor_critic.generate_agent_and_env import create_environment_agent_and_states
from one_step_actor_critic.episode_simulator import run_episode

grid, agent = create_environment_agent_and_states(grid_shape=(4,4),
                                                  initial_position=(0,0),
                                                  final_position=(3,3),
                                                  blocks=[],
                                                  epsilon=0.1,
                                                  agent_actions=[(-1, 0), (1, 0), (0, -1), (0, 1)])


def test_it_runs():
    improved_agent = run_one_step_actor_critic(agent=agent,
                                               environment=grid,
                                               start_state=(0,0),
                                               n_iter=1234,
                                               alpha_th=0.1,
                                               alpha_w=0.1,
                                               gama=0.9)


    print('new w', improved_agent.return_w())
    print('new th', improved_agent.return_th())

    agent.epsilon = 0.01
    state_action_tuple_list, reward_list = run_episode(start_state=(0, 0),
                                                       environment=grid,
                                                       agent=agent)  #

    print('\n number of steps until completion', len(state_action_tuple_list))


