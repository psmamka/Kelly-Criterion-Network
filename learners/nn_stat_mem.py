# Implementation of the Statistical Memory technique
# Continuation of, and parallel to the ε-focus technique
# 
# Statistical Memory Rationale: 
#  * Even when employing ε-focus, the nework has difficulty converging 
#       to the optmial solution due to large variance in outcomes
#  * Instead of recording individual transitions in the memory, we 
#       divide the 2-dimensional state-action (phase) space to tiles
#       (or cells)
#  * With accumulation of experiences, for each cell we keep track of the 
#       average reward, as well as the  number of samples (occurances) 
#       within each cell. 
#  * Here we use non-overlapping cells; for a general discussion of tiling
#       methods consult Sutton & Barto Reinforcement Learning (2018) Ch.9 
#  * In dynamic environments, we can shift the averaging to favor more 
#       recent samples
# 
# 
# Statistical Memory Implementation:
#   * First we initialize the 2D statistical memory grid
#       - A 2D array (matrix) of tuples with (r, n)
#       - r the average reward (utility), n the number of samples for the cell
#       - init values are (0, 0)
#   * With each additional experience/transition:
#       - first calculate the reward (diff of state/capital utilities)
#       - next calculate the memory cell coordinates
#           -- some kind of rounding of (state * N) and (action * M) numbers
#           -- N and M number of rows and columns in the mem matrix erspectively
#           -- Use the projectogram repo indexing code if helpful
#       - then update the cell n and r values
#           -- r → (n * r + r_new) / (n + 1)
#           -- n → n + 1
#       - In each training epoch, build tensor datasets and dataloaders to train
#           the nn
#           -- optionally, use n values as data weights
#   * For q-learning is is not just the current reward that we are interested,
#       - the expected reward in the next step is taken into account
#       - here we detemine the index of the memory row for the next_state
#       - within the row, we can find the highest next reward for all given actions
#       - use the highest next_reward to train the q-nn
#           -- computationally, it would be benefitial to have a column array keeping
#               track of the maximum average reward in each row
#           -- computationally, it might also help to have to matrices for 
#               r (float) and n (int)


#  Statistical Memory 2D Array schema:
#         _ _ _ _ _
#  S     |_|_|_|_|_|    
#  T     |_|_|_|_|_|
#  A     |_|_|_|_|_|    → each cell has (r_ij, n_ij)
#  T     |_|_|_|_|_|    → alternatively, separate matrices for r (float) and n (integer)
#  E     |_|_|_|_|_|
#  S     
# (i)     ACTIONS (j)
# 
# 

#   Integration of statistical memory with epsilon-focus strategy
# 
#   * first: the epsilon action radius need to span multiple cells in the action dimension,
#       in order to achieve proper sampling
#   * second: it makes sense to have epsilonaction centers coincide with memory
#       cell action centers
#   * third: it might be advantageous to use some kind of normal/flat distribution, with σ set to
#       radius and μ to action center, to allow for sufficient sampling from distant actions
#   * fourth: for simplicity sake, we can keep epsilon state centers as memroy cell state centers 

import sys
try:    # for intellisense
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env

import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

from nn_single_trade_q_learn import log_util

Transition = namedtuple('Transition', ('state', 'action', 'reward'))  # for the stat-mem we only need the s/a/r triplet

class StatMemAgent:
    def __init__(self, env, util_func=lambda x: x, rng=np.random.default_rng(), stat_mem_sz=(100, 100),
                    learn_rate=1E-3, st_range=[0.0, 1.0], ac_range=[0.0, 1.0],  eps_foc_init=1.0, 
                    eps_foc_decay=0.999, eps_foc_min=0.5, eps_foc_gran=0.1, discount=1.0, layers_sz=[20, 20], 
                    next_step_lookup=True, epochs_per_episode=1):
        
        self.env = env
        self.util_func = util_func
        
        self.rng = rng

        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair
        self.lr = learn_rate

        # range of state and action values we are training for
        self.st_range = st_range
        self.ac_range = ac_range
        # dividing state and action space to cells (bins). Each cell is represented by the left boundary value
        self.st_bins = np.linspace(start=st_range[0], stop=st_range[1], num=stat_mem_sz[0], endpoint=False)
        self.ac_bins = np.linspace(start=ac_range[0], stop=ac_range[1], num=stat_mem_sz[1], endpoint=False)
        self.cel_sz = (st_range / stat_mem_sz[0], ac_range / stat_mem_sz[1])    # size of individual memory cells

        self.statmem_r = np.zeros(stat_mem_sz)   # stat mem reward matrix (average reward per cell)
        self.statmem_n = np.zeros(stat_mem_sz, dtype=np.uint64)   # stat mem n matrix (number of samples per cell)
        self._statmem_init()                     # stat mem initialization

        self.eps_f_rad = eps_foc_init           # epsilon greedy parameters: radius
        self.eps_f_dec = eps_foc_decay          # decay rate
        self.eps_f_min = eps_foc_min            # minimum radius
        self.eps_f_gran = eps_foc_gran          # granularity for states
        self._epsilon_focus_init()

        self.gamma = discount
        self.next_lookup = next_step_lookup   # q: predictions of next step used for current step optimization
        self.epoc_per_epis = epochs_per_episode

        self._build_qnn(in_sz=self.in_sz, out_sz=self.out_sz, layers_sz=layers_sz)

    def _statmem_init(self):
        # simplest way to initialize the stat mem: have a single sample per each mem cell
        # we choose cell centers (left boundary + half width) as initial samples
        st_arr = self.st_bins + self.cel_sz[0] / 2.0
        ac_arr = self.ac_bins + self.cel_sz[1] / 2.0

        for i, st in enumerate(st_arr):
            for j, ac in enumerate(ac_arr):
                self.env.reset(start_cap=st)
                next_st, _, _ = self.env.step(bet_size= st * ac)
                r = self.state_change_reward(state=st, next_st=next_st)
                self.statmem_r[i, j] = r
                self.statmem_n[i, j] = 1
    
    def state_change_reward(self, state, next_st):
        return self.util_func(next_st) - self.util_func(state)    # utility reward
    
    def _epsilon_focus_init(self, verbose=False):
        # number of epsilon centers to keep track of: (st_max - st_min) / epsilon_granularity + 1 
        # (+1 because exclusive of boundaries)
        num_eps_states = int((self.st_range[1] - self.st_range[0]) / self.eps_f_gran + 1)
        self.eps_f_states = np.linspace(self.st_range[0], self.st_range[1], num_eps_states, endpoint= True)
        avg_action = (self.ac_range[1] - self.ac_range[0]) / 2.0    # middle point for possible actions used for init
        self.eps_ac_centers = np.ones(num_eps_states) * avg_action
        if verbose:
            print(f"eps_f granular states: {self.eps_f_states} \neps_f action centers: {self.eps_ac_centers}")

    def build_qnn(self, in_sz=2, out_sz=1, layers_sz=[20, 20]):
        layer_list = []
        layer_sizes = zip([in_sz] + list(layers_sz), list(layers_sz) + [out_sz])

        for idx, (num_in, num_out) in enumerate(layer_sizes):
            layer_list.append(nn.Linear(num_in, num_out))
               # rectifier layer: ReLU() | LeakyReLU(negative_slope=0.1) | Sigmoid() | 
            if idx < len(layers_sz): layer_list.append(nn.ReLU())

        self.model = nn.Sequential(*layer_list) # print(self.model)

        self.loss_fn = nn.MSELoss()   # MSELoss() | L1Loss() | CrossEntropyLoss() | BCELoss()
        # reduction = 'mean' (default) | 'none' | 'sum'
        # optimizer:s SGD | Adam
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=0.0)
        
    def remember(self, tr):
        i = self.get_state_mem_index(tr.state)      # consider refactoring to one method 
        j = self.get_action_mem_index(tr.action)    # ←
        old_r, old_n = self.statmem_r[i, j], self.statmem_n[i, j]
        
        # new_avg_r = (n * old_avg_r + new_r) / (n + 1)   ← can use a different formula to consider recency
        self.statmem_r[i, j] = (old_n * old_r + tr.reward) / (old_n + 1)
        self.statmem_n[i, j] = old_n + 1
    
    def get_state_mem_index(self, st):
        # index = floor( (x - min_x) / x_step)  → floor returns float, cast to uint
        st_idx = np.floor((st - self.st_range[0]) / self.cel_sz[0]).astype(np.uint64)
        return st_idx
    
    def get_action_mem_index(self, ac):
        ac_idx = np.floor((ac - self.ac_range[0]) / self.cel_sz[1]).astype(np.uint64)
        return ac_idx

    def recall(self):
        # for stat-mem, we need all cells. Just return linearized avg_r and n martices
        return self.statmem_r.flatten(), self.statmem_n.flatten()

    def select_action(self, state, verbose=False):
        # Here epsilon-focus and statistical memory need to be integrated
        # ...
        return

# ===== Execution =====
if __name__ == '__main__':
    rng = np.random.default_rng()
    torch.manual_seed(1)

    # betting env properties
    prob_arr = np.array([0.3, 0.7]) #
    outcome_arr = np.array([0.0, 2.0])
    # validation
    num_st, num_ac = 10, 21
    # validation data grid
    val_st_range = np.array([0.1, 1.0])
    val_ac_range = np.array([0.1, 0.9])
    st_minmax = np.array([0.0, 2.0])
    util_func = lambda x: log_util(x, x_reg=1E-3)
    # util_func = lambda x: x
    normalize_uf = True     # True: Normalize utility function to [-1 +1] range | False: use raw util func
    if normalize_uf: 
        n_util_func = StatMemAgent.normalize_util_func(util_func, minmax=st_minmax, num=11, verbose=True)
    else:
        n_util_func = util_func