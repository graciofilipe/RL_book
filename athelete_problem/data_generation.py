import numpy as np
import pickle
import pandas as pd


def is_state_final(state):
    return state[0] >= 6 and state[1] >= 6

def get_next_state_from_action(state, action):

    if state[2] >= 2 and action != 'rest':

        r = np.random.rand()
        if r < 0.8:
            increment = [0, 0, 1]
        else:
            increment = [0, 0, 2]

        new_state = [state[i] + increment[i] for i in [0, 1, 2]]
        new_state[0] = np.min([np.max([new_state[0], 0]), 7])
        new_state[1] = np.min([np.max([new_state[1], 0]), 7])
        new_state[2] = np.min([np.max([new_state[2], 0]), 7])

        return new_state

    elif action == 'strength_1':
        r = np.random.rand()
        if r < 0.7:
            increment = [1, 0, 1]
        elif r > 0.7 and r < 0.9:
            increment = [0, 0, 1]
        elif r >= 0.9:
            increment = [1, 0, 0]


    elif action == 'strength_2':
        if state[2] >= 1:
            r = np.random.rand()
            if r < 0.9:
                increment = [0, 0, 2]
            elif r > 0.9:
                increment = [0, 0, 1]
        else:
            r = np.random.rand()
            if r < 0.8:
                increment = [2, 0, 1]
            elif r > 0.8:
                increment = [1, 0, 1]


    elif action == 'flexibility_1':
        r = np.random.rand()
        if r < 0.7:
            increment = [0, 1, 1]
        elif r > 0.7 and r < 0.9:
            increment = [0, 0, 1]
        elif r >= 0.9:
            increment = [0, 1, 0]

    elif action == 'flexibility_2':
        if state[2] >= 1:
            r = np.random.rand()
            if r < 0.85:
                increment = [0, 0, 2]
            elif r > 0.85:
                increment = [0, 0, 1]
        else:
            r = np.random.rand()
            if r < 0.8:
                increment = [0, 2, 1]
            elif r > 0.8 and r < 0.93:
                increment = [0, 1, 1]
            else:
                increment = [1, 2, 1]

    elif action == 'rest':
        r = np.random.rand()
        if r < 0.8:
            increment = [0, 0, -2]
        else:
            increment = [0, 0, -1]

    new_state = [state[i] + increment[i] for i in [0, 1, 2]]
    new_state[0] = np.min([np.max([new_state[0], 0]), 7])
    new_state[1] = np.min([np.max([new_state[1], 0]), 7])
    new_state[2] = np.min([np.max([new_state[2], 0]), 7])

    return new_state

def generate_data(n_trajectories):

    episode_states =[]
    episode_actions = []
    for n in range(n_trajectories):
        state_list = [[0, 0, 0]]
        action_list = []
        while not is_state_final(state_list[-1]):
            if state_list[-1][2] >= 4:
                r = np.random.rand()
                if r < 0.8:
                    action = 'rest'
                else:
                    action = np.random.choice(['strength_1', 'strength_2', 'flexibility_1', 'flexibility_2', 'rest'])
            else:
                action = np.random.choice(['strength_1', 'strength_2', 'flexibility_1', 'flexibility_2', 'rest'])
            action_list.append(action)
            new_state = get_next_state_from_action(state_list[-1], action)
            state_list.append(new_state)
        episode_actions.append(action_list)
        episode_states.append(state_list)

    lens = list(map(len, episode_states))
    print('random workouts took average of ', np.mean(lens), 'steps to reach goal',
          'with std of', np.std(lens))

    filehandler = open('episode_actions.pkl', 'wb')
    pickle.dump(episode_actions, filehandler)

    filehandler = open('episode_states.pkl', 'wb')
    pickle.dump(episode_states, filehandler)


    # save as csvs:
    df_list = []
    for traj in range(n_trajectories):

        df_list.append(pd.DataFrame(data={'action': episode_actions[traj] + ['NA'],
                                          'state': episode_states[traj],
                                          'time': list(range(len(episode_states[traj]))),
                                          'user':'user_'+str(traj+1)}
                                    )
                       )

    df = pd.concat(df_list)
    df.to_csv('data.csv', index=False)

    # import ipdb; ipdb.set_trace()
    x=1

if __name__ == '__main__':
    generate_data(60000)