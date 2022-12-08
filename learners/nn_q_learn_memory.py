# A neural network with a memroy queue (q-learning) for optimal investment ratio
# The general layout is roughly parallel to that of Raschka et al. Python Machine Learning, 
# PyTorch Edition 2022 Ch.19 DQN examples. Due to continuous action, the output layer in
# our network is just the q value, with multiple inputs (state/action) tried to find the
# q-optimal action.
# 
# 
# General Steps:
# - Initialize Neural Network
# - Initialize the memory double-ended queue
# - Train the network for various initial states
#   -- epsilon-greedy, with decaying epsilon
# - plot training history
# - plot performance, compare to theoretical util

import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque
import matplotlib.pyplot as plt

from nn_single_trade_q_learn import \
    log_util, get_determ_reward_y, get_stoch_next_state, get_state_change_reward, select_action

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class QInvestAgent:
    def __init__(self, mem_size=int(1E6), learn_rate=1E-3, eps_init=1.0, eps_decay=0.999, eps_min=0, layers_sz=[20, 20]):
        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair

        self.memory = deque(maxlen=mem_size)
        self.lr = learn_rate

        self.eps = eps_init         # epsilon greedy parameters
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self._build_qnn(in_sz=self.in_sz, out_sz=self.out_sz, layers_sz=layers_sz)
    
    def _build_qnn(self, in_sz=2, out_sz=1, layers_sz=[20, 20]):
        layer_list = []
        layer_sizes = zip([in_sz] + list(layers_sz), list(layers_sz) + [out_sz])

        for idx, (num_in, num_out) in enumerate(layer_sizes):
            layer_list.append(nn.Linear(num_in, num_out))
            if idx < len(layers_sz): layer_list.append(nn.ReLU())

        self.model = nn.Sequential(*layer_list) # print(self.model)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    
    def remember(self, tr):
        self.memory.append(tr)
    



if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)

    prob_arr = np.array([0.3, 0.7]) # [0.4, 0.6]
    outcome_arr = np.array([0.0, 2.0])
    st_range = np.array([0.01, 1.0])
    ac_range = np.array([0, 0.99])
    st_minmax = np.array([0.01, 2.0])
    util_func = lambda x: log_util(x, x_reg=1E-5)
    epsilon = 1.0

    agent = QInvestAgent()

