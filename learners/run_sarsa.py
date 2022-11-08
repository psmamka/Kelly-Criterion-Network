import numpy as np
import sys
try:    # for intellisense
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env
from sarsa import SarsaLearner
import matplotlib.pyplot as plt

EPISODES = 5000

if __name__ == '__main__':
    bet_env = betting_env.BettingEnvBinary(win_pr=0.6, loss_pr=0.4, win_fr=1.0, loss_fr=1.0, 
                                        start_cap=10, max_cap=1000, min_cap=1, max_steps=20)

    sarsa_agent = SarsaLearner(bet_env, learning_rate=1E-9, discount_factor=1.0,
                            epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05,
                            q_reg = 1E6, w_reg=10)
    
    state = bet_env.reset()

    final_states = np.zeros(EPISODES)

    for e in range(EPISODES):
        state = bet_env.reset(np.random.uniform(1, 100))
        states_list = [state]
        action = sarsa_agent.choose_action(state)

        for i in range(bet_env.max_steps):
            # print(i, state, action)
            next_state, next_reward, terminal = bet_env.step(action)
            states_list.append(next_state)

            if terminal or i == bet_env.max_steps - 1:
                final_states[e] = next_state
                sarsa_agent.update_weights_terminal(state, action, next_reward)
                break
            else:
                reward = next_reward
                next_action = sarsa_agent.choose_action(next_state)
                sarsa_agent.update_weights(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action

        # plot states progression in one episode
        # print(states_list)
    
    # print final states for all episodes
    print(np.array2string(final_states, formatter={'float_kind':lambda x: "%.2f" % x})) 
    print("Weights: ", f"{sarsa_agent.w_01:.2f}, {sarsa_agent.w_10:.2f}, {sarsa_agent.w_02:.2f}, {sarsa_agent.w_11:.2f}, {sarsa_agent.w_20:.2f}")

    # plot training results:
    s_arr = np.linspace(start=1, stop=100, num=100, endpoint=True)
    a_arr = np.zeros(len(s_arr))
    for idx, s in enumerate(s_arr):
        a_arr[idx] = sarsa_agent.choose_action(s_arr[idx])
    
    plt.figure()
    plt.plot(s_arr, a_arr)
    plt.ylim((-5, 100))
    plt.show()

