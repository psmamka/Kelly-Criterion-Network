# The Semi-Gradient Sarsa technique with approximation of the action-value function to achieve the 
# optimal betting performance.
# 
# For a theoretical treatment, see: R. Sutton, A. Barto - Reinforcement Learning, An Introduction (2018), Ch. 10
# Implementation and method naming are roughly similar to those in S. Raschka, et al: Machine Learning with PyTorch (2022) Ch. 19

# For a parametric approximation of the action-value function, we use the following quadratic form for
# state s and action a and weights vector w:

# q(s, a, w) = w_10 * s + w_01 * a + w_20 * s^2 + w_11 * s * a + w_02 * a ^ 2

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
# where Grad_w(q) indicated the gradient of q(s, a, w) with respect to w
# 
# To allow exploration, epsilon-greedy action selection is used.
# 

import numpy as np

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

    def choose_action(self, state):
        # 
        if np.random.uniform(0, 1) <= self.eps:
            action = np.random.uniform(0, state)
            return action

        a_opt = (self.w_11 * state + self.w_01) / (-2 * self.w_02)

        if a_opt < 0:
            return 0
        elif a_opt > state:
            return state
        else:
            return a_opt
        
    