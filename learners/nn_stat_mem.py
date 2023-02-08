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

class StatMemAgent:
    def __init__(self, env, util_func=lambda x: x, rng=np.random.default_rng(), stat_mem_sz=(20, 20),
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
        self.bin_sz = (st_range / stat_mem_sz[0], ac_range / stat_mem_sz[1])

        self.statmem_r = np.zeros(stat_mem_sz)   # stat mem reward matrix (average reward per cell)
        self.statmem_n = np.zeros(stat_mem_sz, dtype=np.uint64)   # stat mem n matrix (number of samples per cell)
        self._sm_init()                     # stat mem initialization

        self.eps_f_rad = eps_foc_init           # epsilon greedy parameters: radius
        self.eps_f_dec = eps_foc_decay          # decay rate
        self.eps_f_min = eps_foc_min            # minimum radius
        self.eps_f_gran = eps_foc_gran          # granularity for states
        self._epsilon_focus_init()

        self.gamma = discount
        self.next_lookup = next_step_lookup   # q: predictions of next step used for current step optimization
        self.epoc_per_epis = epochs_per_episode

        self._build_qnn(in_sz=self.in_sz, out_sz=self.out_sz, layers_sz=layers_sz)

    def _sm_init(self):
        # simplest way to initialize the stat mem: have a single sample per each mem cell
        # we choose cell centers (left boundary + half width) as a smaple

        st_arr = self.st_bins + self.bin_sz[0] / 2.0
        ac_arr = self.ac_bins + self.bin_sz[1] / 2.0

        for i, st in enumerate(st_arr):
            for j, ac in enumerate(ac_arr):
                self.env.reset(start_cap=st)
                next_st, _, _ = self.env.step(bet_size= st * ac)
                r = self.state_change_reward(state=st, next_st=next_st)
                self.statmem_r[i, j] = r
                self.statmem_n[i, j] = 1
        
