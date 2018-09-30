from prioritised_sweep_8.generate_agent_and_eng import create_environment_agent_and_states
from prioritised_sweep_8.sweeper import run_sweep

environment, agent = create_environment_agent_and_states(grid_shape=(6, 6),
                                                         initial_position=(0, 0),
                                                         final_position=(5, 5),
                                                         epsilon=0.1)

new_agent = run_sweep(environment=environment,
                      agent=agent,
                      alpha=0.5,
                      gama=0.9,
                      n_iter=10,
                      theta=0.1)

