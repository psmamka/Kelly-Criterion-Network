# The Semi-Gradient Sarsa technique with approximation of the action-value function to achieve the 
# optimal betting performance.
# 
# For a theoretical treatment, see: R. Sutton, A. Barto - Reinforcement Learning, An Introduction (2018), Ch. 10
# Implementation and method namings are roughly similar to those in S. Raschka, et al: Machine Learning with PyTorch (2022) Ch. 19
# 

# For a parametric approximation of the action-value function, we use the following quadratic form for
# state s and action a and weights vector w:

# q(s, a, w) = w_01 * a + w_10 * s + w_02 * a ^ 2 + w_11 * s * a + w_20 * s^2 

# To find the best action a at each state s, we calculate the q partial derivate w.r.t. a:
# 
# dq(s, a, w)/da =  w_01 + w_11 * s + 2 * w_02 * a
# 
# Setting the derivative to zero, we get the extremum position for a:
# 
# a = -1 * (w_01 + w_11 * s) / (2 * w_02)
# 
# The extremum a is chosen as the policy value, assuming that min/max limits of a (0 and s) result in inferior 
# performance. If the extremum falls outside the limits, it is replaced with the respective limit.
# 
# The weights vector w = [w_01, w_10, w_02, w_11, w20] is updated in each time step t as following:
# 
# w_(t+1) = w_t + learning_rate * [R_(t+1) + discount_factor * q(s_(t+1), a_(t+1), w_t) - q(s_t, a_t, w_t)] * Grad_w(q(s_t, a_t, w_t))
# 
# where Grad_w(q) indicated the gradient of q(s, a, w) with respect to w:
# 
# for quadratic form (lexical ordering of indices 01 to 20): Grad_w(q) = [a, s, a^2, sa, s^2]
# 
# To allow exploration, epsilon-greedy action selection is used.
# 

import numpy as np
from .. import betting_env

class SarsaLearner:
    def __init__(self, env, learning_rate=0.1, discount_factor=1.0,
                    epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.001):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon
        self.eps_dec = epsilon_decay
        self.eps_min = epsilon_min

        self._initialize_weights()

    def _initialize_weights(self):
        # w vector: [w_01, w_10, w_02, w_11, w_20]
        self.w_01, self.w_10, self.w_02, self.w_11, self.w_20 =  np.random.uniform(-1.0, 1.0, 5)

    def _q_val(self, s, a):
        # quadratic form in state (s) and action (a)
        q = self.w_01 * a + self.w_10 * s + self.w_02 * a * a +  self.w_11 * a * s + self.w_20 * s * s
        return q

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.uniform(0, 1) <= self.eps:
            action = np.random.uniform(0, state)
            return action

        a_opt = (self.w_11 * state + self.w_01) / (-2 * self.w_02)
        # action (betting amount) must be between 0 and full capital (state):
        if a_opt < 0:
            return 0
        elif a_opt > state:
            return state
        else:
            return a_opt
        
    def update_weights_terminal(self, s, a, reward):
        # terminal state update
        # w → w + alpha * [R_(t+1) - q(s(t), a(t), w(t))] * Grad_w(q(s, a, w))
        # Grad_w(q) = [a, s, a^2, s.a, s^2]
        cor = self.lr * (reward - self._q_val(s, a))
        [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] += cor * np.array([a, s, a * a, s * a, s * s])

    def update_weights(self, s, a, reward, next_s, next_a):
        # non-terminal update
        # w → w + alpha * [R_(t+1) + gamma * q(s(t+1), a(t+1), w(t)) - q(s(t), a(t), w(t))] * Grad_w
        cor = self.lr * (reward + self.gamma * self._q_val(next_s, next_a) - self._q_val(s, a))
        [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] += cor * np.array([a, s, a * a, s * a, s * s])

        
    
EPISODES = 100

if __name__ == '__main__':
    bet_env = betting_env.BettingEnvBinary(win_pr=0.6, loss_pr=0.4, win_fr=1.0, loss_fr=1.0, 
                                        start_cap=100, max_cap=1E6, min_cap=1, max_steps=100)

    sarsa_agent = SarsaLearner(bet_env, learning_rate=0.1, discount_factor=1.0,
                            epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.001)
    
    state = bet_env.reset()

    final_states = np.zeros(EPISODES)

    for e in range(EPISODES):
        states = np.zeros(bet_env.max_steps)
        state = bet_env.reset()
        action = sarsa_agent.choose_action(state)

        for i in range(bet_env.max_steps):
            next_state, next_reward, terminal = bet_env.step(action)
            states.append(next_state)

            if terminal or i == bet_env.max_steps - 1:
                final_states[e] = state
                sarsa_agent.update_weights_terminal(state, action, reward)
                break
            else:
                reward = next_reward
                next_action = sarsa_agent.choose_action(state)
                sarsa_agent.update_weights(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action

        # plot states progression in one episode
    
    # plot final states for all episodes
