from prioritised_sweep_8.env_class import Environment


# return_state_and_reward_post_action

# set_state

def test_set_and_get_state():

    grid_shape = (2,2)
    initial_position = (0,0)
    final_position = (1,1)
    blocks = []

    grid = Environment(grid_shape=grid_shape,
                       initial_position=initial_position,
                       final_position=final_position,
                       blocks=blocks)

    assert grid.get_state() == initial_position

    new_state = (1, 0)
    grid.set_state(new_state=new_state)
    assert grid.get_state() == new_state



def test_return_state_and_reward_post_action():
    grid_shape = (2, 2)
    initial_position = (0, 0)
    final_position = (1, 1)
    blocks = []

    grid = Environment(grid_shape=grid_shape,
                       initial_position=initial_position,
                       final_position=final_position,
                       blocks=blocks)

    action = (1, 0)
    next_state, reward, terminal_flag = grid.return_state_and_reward_post_action(action=action)
    assert next_state == (1, 0)
    assert terminal_flag == False
    assert reward == -1

    action = (-1, 0)
    next_state, reward, terminal_flag = grid.return_state_and_reward_post_action(action=action)
    assert next_state == (0, 0)
    assert terminal_flag == False
    assert reward == -1

    action = (0, 1)
    next_state, reward, terminal_flag = grid.return_state_and_reward_post_action(action=action)
    assert next_state == (0, 1)
    assert terminal_flag == False
    assert reward == -1

    action = (0, -1)
    next_state, reward, terminal_flag = grid.return_state_and_reward_post_action(action=action)
    assert next_state == (0, 0)
    assert terminal_flag == False
    assert reward == -1

    action = (1, 1)
    next_state, reward, terminal_flag = grid.return_state_and_reward_post_action(action=action)
    assert next_state == (1, 1)
    assert terminal_flag == True
    assert reward == 0

