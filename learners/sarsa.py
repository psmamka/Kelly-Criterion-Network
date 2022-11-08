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
# The second class has a simpler linear form for q(s, a, w) =  w_01 * a + w_10 * s

import numpy as np
import sys
try:
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env

class SarsaLearnerQuadratic:
    '''Quadratic Implementation of Semigradient Sarsa where q is approximated using a second order polynomial in 
    s and a: 
    q(s, a, w) = w_01 * a + w_10 * s + w_02 * a ^ 2 + w_11 * s * a + w_20 * s^2'''
    def __init__(self, env, learning_rate=1E-8, discount_factor=1.0,
                    epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.001,
                    q_reg=1E6, w_reg=1E3):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon
        self.eps_dec = epsilon_decay
        self.eps_min = epsilon_min
        self.q_reg = q_reg
        self.w_reg = w_reg

        self._initialize_weights()

    def _initialize_weights(self):
        # w vector: [w_01, w_10, w_02, w_11, w_20]
        # self.w_01, self.w_10, self.w_02, self.w_11, self.w_20 =  np.random.uniform(-0.01, 0.01, 5)
        self.w_01, self.w_10, self.w_02, self.w_11, self.w_20 = np.zeros(5)

    def _q_val(self, s, a):
        # quadratic form in state (s) and action (a)
        q = self.w_01 * a + self.w_10 * s + self.w_02 * a * a +  self.w_11 * a * s + self.w_20 * s * s
        if np.isnan(q) or np.isinf(q):
            print("overflow problem in calculating q:")
            print(f"w_01: {self.w_01}    w_10: {self.w_10}    w_02: {self.w_20}    w_11: {self.w_11}    w_20: {self.w_20} ")
        q = self._regularize_q(q)
        return q

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.uniform(0, 1) <= self.eps:
            action = np.random.uniform(0, state)
        else:
            action = self._optimal_action(state)
        
        # action (betting amount) must be between 0 and full capital (state):
        if action < 0: action = 0
        if action > state: action = state
        
        return action

    def _optimal_action(self, state):
        optimal_action = (self.w_11 * state + self.w_01) / (-2 * self.w_02) # optimum a_opt from derivative = 0

        q_opt_a = self._q_val(state, optimal_action)
        q_a_0 = self._q_val(state, 0)   #       q(s, 0) for a = 0, i.e. betting 0
        q_a_s = self._q_val(state, state)   #   q(s, s) for acion = state, i.e. betting full

        return max([(0, q_a_0), (state, q_a_s), (optimal_action, q_opt_a)], key=lambda t: t[1])[0]  # 
        
    def update_weights_terminal(self, s, a, reward):
        # terminal state update
        # w → w + alpha * [R_(t+1) - q(s(t), a(t), w(t))] * Grad_w(q(s, a, w))
        # Grad_w(q) = [a, s, a^2, s.a, s^2]
        cor = self.lr * (reward - self._q_val(s, a))
        [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] = \
            [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] + cor * np.array([a, s, a * a, s * a, s * s])
        self._regularize_weights()
        self._adjust_epsilon()

    def update_weights(self, s, a, reward, next_s, next_a):
        # non-terminal update
        # w → w + alpha * [R_(t+1) + gamma * q(s(t+1), a(t+1), w(t)) - q(s(t), a(t), w(t))] * Grad_w
        cor = self.lr * (reward + self.gamma * self._q_val(next_s, next_a) - self._q_val(s, a))
        [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] = \
            [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] + cor * np.array([a, s, a * a, s * a, s * s])
        
        self._regularize_weights()
        self._adjust_epsilon()

    def _regularize_weights(self):
        # for elm in [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20]:
        #     if elm > lim: elm = lim
        #     if elm < -lim: elm = -lim
        arr = [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20]
        [self.w_01, self.w_10, self.w_02, self.w_11, self.w_20] = \
            list( map(lambda x: max(min(x, self.w_reg), -self.w_reg), arr) )

    def _regularize_q(self, q_in):
        return max(min(q_in, self.q_reg), -self.q_reg)
        
    def _adjust_epsilon(self):
        if self.eps > self.eps_min: self.eps *= self.eps_dec
        

class SarsaLearnerLinear:
    '''q(s, a, w) is approximated as a linear function of s and a: q(s, a, w) = w_01 * a + w_10 * s'''
    def __init__(self, env, learning_rate=0.1, discount_factor=1.0,
                    epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.001,
                    q_reg=1E6, w_reg=1E3):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon
        self.eps_dec = epsilon_decay
        self.eps_min = epsilon_min
        self.q_reg = q_reg
        self.w_reg = w_reg

        self._initialize_weights()

    def _initialize_weights(self):
        self.w_01, self.w_10 = [0, 0]

    def _q_val(self, s, a):
        # quadratic form in state (s) and action (a)
        q = self.w_01 * a + self.w_10 * s
        if np.isnan(q) or np.isinf(q):
            print("overflow problem in calculating q:")
            print(f"w_01: {self.w_01}    w_10: {self.w_10}")
        
        q = self._regularize_q(q)
        return q

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.uniform(0, 1) <= self.eps:
            action = np.random.uniform(0, state)
        else:
            action = self._optimal_action(state)
        
        # action (betting amount) must be between 0 and full capital (state):
        if action < 0: action = 0
        if action > state: action = state
        
        return action

    def _optimal_action(self, state):
        # for a linear function, we only need to compare the two extremes of a=0 and a=s. i.e. binary action
        q_a_0 = self._q_val(state, 0)   #       q(s, 0) for a = 0, i.e. betting 0
        q_a_s = self._q_val(state, state)   #   q(s, s) for acion = state, i.e. betting full

        return max([(0, q_a_0), (state, q_a_s)], key=lambda t: t[1])[0]  # 
        
    def update_weights_terminal(self, s, a, reward):
        # terminal state update
        # w → w + alpha * [R_(t+1) - q(s(t), a(t), w(t))] * Grad_w(q(s, a, w))
        # Grad_w(q) = [a, s, a^2, s.a, s^2]
        cor = self.lr * (reward - self._q_val(s, a))
        [self.w_01, self.w_10] =  [self.w_01, self.w_10] + cor * np.array([a, s])
        self._regularize_weights()
        self._adjust_epsilon()

    def update_weights(self, s, a, reward, next_s, next_a):
        # non-terminal update
        # w → w + alpha * [R_(t+1) + gamma * q(s(t+1), a(t+1), w(t)) - q(s(t), a(t), w(t))] * Grad_w
        cor = self.lr * (reward + self.gamma * self._q_val(next_s, next_a) - self._q_val(s, a))
        [self.w_01, self.w_10] = [self.w_01, self.w_10] + cor * np.array([a, s])
        
        self._regularize_weights()
        self._adjust_epsilon()

    def _regularize_weights(self):
        # arr = [self.w_01, self.w_10]
        [self.w_01, self.w_10] = list( map(lambda x: max(min(x, self.w_reg), -self.w_reg), [self.w_01, self.w_10]) )

    def _regularize_q(self, q_in):
        return max(min(q_in, self.q_reg), -self.q_reg)
        
    def _adjust_epsilon(self):
        if self.eps > self.eps_min: self.eps *= self.eps_dec
