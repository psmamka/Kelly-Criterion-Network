import numpy as np
import sys
try:    # for intellisense
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env
from sarsa import SarsaLearner
    
EPISODES = 20

if __name__ == '__main__':
    bet_env = betting_env.BettingEnvBinary(win_pr=0.6, loss_pr=0.4, win_fr=1.0, loss_fr=1.0, 
                                        start_cap=10, max_cap=100, min_cap=1, max_steps=20)

    sarsa_agent = SarsaLearner(bet_env, learning_rate=0.01, discount_factor=1.0,
                            epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.001)
    
    state = bet_env.reset()

    final_states = np.zeros(EPISODES)

    for e in range(EPISODES):
        state = bet_env.reset()
        states_list = [state]
        action = sarsa_agent.choose_action(state)

        for i in range(bet_env.max_steps):
            print(i, state, action)
            next_state, next_reward, terminal = bet_env.step(action)
            states_list.append(next_state)

            if terminal or i == bet_env.max_steps - 1:
                final_states[e] = state
                sarsa_agent.update_weights_terminal(state, action, next_reward)
                break
            else:
                reward = next_reward
                next_action = sarsa_agent.choose_action(next_state)
                sarsa_agent.update_weights(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action

        # plot states progression in one episode
        print(states_list)
    
    # plot final states for all episodes
    print(final_states)
