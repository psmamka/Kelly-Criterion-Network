# Train and evaluate quadratic sarsa learner, optimizing for highest rate of return: reward = return / capital
# The theoretical optimum is provided by the Kelly Criterion:
# Optimal Betting Fraction = Win_probability / Loss_fraction   -   Loss_probability / Win_fraction
# See explanationa and plots from binary_bet_theory.py

import numpy as np
import sys
try:    # for intellisense
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env
from sarsa import SarsaLearnerQuadratic
import matplotlib.pyplot as plt


bet_env = betting_env.BettingEnvBinary(win_pr=0.6, loss_pr=0.4, win_fr=1.0, loss_fr=1.0, 
                                        start_cap=10, max_cap=100, min_cap=1, max_steps=20)

def train_sarsa_quadratic_rate(bet_env):
    EPISODES = 1000

    sarsa_agent = SarsaLearnerQuadratic(bet_env, learning_rate=1E-7, discount_factor=1.0,
                            epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05,
                            q_reg = 1E5, w_reg=100)
    
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
            next_reward /= state    # <==== reward set to return per capital

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
    print("Weights: ", f"{sarsa_agent.w_02:.2f}, {sarsa_agent.w_11:.2f}, {sarsa_agent.w_20:.2f}")
    return sarsa_agent

def plot_sarsa_action(sarsa_agent, title=None):
    # plot training results:
    s_arr = np.linspace(start=1, stop=100, num=100, endpoint=True)
    a_arr = np.zeros(len(s_arr))
    for idx, s in enumerate(s_arr):
        a_arr[idx] = sarsa_agent.choose_action(s_arr[idx])
    
    plt.figure()
    plt.plot(s_arr, a_arr)
    plt.ylim((-5, s_arr[-1] * 1.05))
    if title is not None: plt.title(title)
    plt.show()

agent_quadratic = train_sarsa_quadratic_rate(bet_env)
plot_sarsa_action(agent_quadratic, title="quadratic sarsa")

