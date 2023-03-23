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
        # random number generator
        self.rng = rng
        # statistical memory grid size:
        self.stat_mem_sz = stat_mem_sz
        self.stat_mem_len = stat_mem_sz[0] * stat_mem_sz[1]
        # neural network in/out 
        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair
        self.lr = learn_rate

        # range of state and action values we are training for
        self.st_range = st_range
        self.ac_range = ac_range
        # dividing state and action space to cells (bins). Each cell is represented by the bin <center> value
        self.cel_sz = (st_range[1] / stat_mem_sz[0], ac_range[1] / stat_mem_sz[1])    # the 2D size of individual memory cells
        self.st_bins = np.linspace(start=st_range[0], stop=st_range[1], num=stat_mem_sz[0], endpoint=False) + \
                            0.5 * self.cel_sz[0]
        self.ac_bins = np.linspace(start=ac_range[0], stop=ac_range[1], num=stat_mem_sz[1], endpoint=False) + \
                            0.5 * self.cel_sz[1]

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
        st_arr = self.st_bins # + self.cel_sz[0] / 2.0  
        ac_arr = self.ac_bins # + self.cel_sz[1] / 2.0

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
        # (+1 because inclusive of boundaries)
        # Note: it might make sense to have the same number epsilon centers and stat mem states
        num_eps_states = int((self.st_range[1] - self.st_range[0]) / self.eps_f_gran + 1)
        self.eps_f_states = np.linspace(self.st_range[0], self.st_range[1], num_eps_states, endpoint= True)
        avg_action = (self.ac_range[1] - self.ac_range[0]) / 2.0    # middle point for possible actions used for init
        self.eps_ac_centers = np.ones(num_eps_states) * avg_action
        if verbose:
            print(f"eps_f granular states: {self.eps_f_states} \neps_f action centers: {self.eps_ac_centers}")

    def _build_qnn(self, in_sz=2, out_sz=1, layers_sz=[20, 20]):
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
        # index = floor( (x - min_x) / x_step )  → floor returns float, cast to uint
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
        # One approach is to use a normal/gaussian dist with stan.dev of epsilon
        #   - for values returned outside the admissible action range, choose one inside uniformly
        # Another approach is to repurpose the uniform epsilon interval for stat mem
        #   - here values are chosen uniformly within an epsilon radius interval
        #   - the problem is that the values outside the radius at some point never get sampled again
        #   - so we may miss optimal solutions if epsilon shrinks too fast

        i = self.get_state_mem_index(state)
        # 1. find the action center (optimal bin)
        j = np.argmax(self.statmem_r[i, :])     # get the optimal action mem index
        ac_center = self.ac_bins[j] # + 0.5 * self.cel_sz[1]
        # 2. set the focus boundaries: truncate to admissible action boundaries
        ac_min = max(ac_center - self.eps_f_rad, self.ac_range[0])   
        ac_max = min(ac_center + self.eps_f_rad, self.ac_range[1])
        # 3. uniform selection within the epsilon range
        # Note: alternatively we can do uniform discrete selection from bins within the range 
        action = self.rng.uniform(low=ac_min, high=ac_max)

        if verbose: print(f"select_action test: state: {state} | s_index: {i} | ac_min: {ac_min} | ac_max: {ac_max} | action: {action}")
        return action
    
    def _update_eps_foc(self, verbose=False):    # update epsilon focus: only the action radius
        old_rad = self.eps_f_rad
        # update epsilon radius
        if self.eps_f_rad > self.eps_f_min: self.eps_f_rad *= self.eps_f_dec

        if verbose: print(f"eps-foc radius update done, before: {old_rad} | after: {self.eps_f_rad}")
        
    def learn_step(self):
        # Here the `recall samples` are the stat-mem cells, propely weighted
        # recall_inputs, recall_targets = [], []
        recall_len = self.stat_mem_len

        if self.next_lookup: # the q method: use predictions for the next steps
            print("Needs to be done later!!!")
        else:
            recall_st, recall_ac = np.meshgrid(self.st_bins, self.ac_bins, indexing='ij')   # check the indexing
            recall_inputs = np.hstack((recall_st.reshape(recall_len, 1), recall_ac.reshape(recall_len, 1)))
            recall_targets = self.statmem_r.reshape(recall_len)

            # for now not using the multiplicity (statmem_n)
            train_dataset = TensorDataset(torch.tensor(recall_inputs, dtype=torch.float32), 
                                            torch.tensor(recall_targets, dtype=torch.float32))
            train_dl = DataLoader(train_dataset, batch_size=len(recall_targets), shuffle=True)
        
        for epoch in range(self.epoc_per_epis):
            for x_b, y_b in train_dl:   # bathces
                # recall_pred = self.model(torch.tensor(recall_inputs, dtype=torch.float32)).squeeze()  # [0]
                # loss = self.loss_fn(recall_pred, torch.stack(recall_targets, dim=0))
                pred = self.model(x_b)[:, 0]
                loss = self.loss_fn(pred, y_b.squeeze())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return loss.item()
    
    def train_nn(self, num_epis=int(1E5), epis_prog=int(1E3)):
        start_t = time.time()
        # nn.init.xavier_uniform_(self.model[0].weight)
        for layer in self.model:    # initialize linear layers
            if type(layer) == nn.Linear: nn.init.xavier_uniform_(layer.weight)

        self.train_loss_hist = np.zeros(num_epis // epis_prog)
        self.valid_loss_hist = np.zeros(num_epis // epis_prog)

        if self.x_valid is None or self.y_valid is None:
            raise Exception("validation data need to be initialized")

        for epis in range(num_epis):
            # single episode logic:
            # (1) initialize env state
            # (2) select action
            # (3) get reward
            # (4) remember transition
            # (5) train network
            # (6) update epsilon-focus
            s = self.rng.uniform(low=self.st_range[0], high=self.st_range[1])
            self.env.reset(start_cap = s)
            
            a = self.select_action(state = s)
            next_s, _, _ = self.env.step(bet_size= s * a)
            r = self.state_change_reward(state=s, next_st=next_s)
            tr = Transition(state=s, action=a, reward=r) #, next_state=next_s)
            self.remember(tr)
            train_loss = self.learn_step()
            self._update_eps_foc()    # <== update epsilon focus radii

            if epis % epis_prog == 0:   # progress report logic
                self.train_loss_hist[epis // epis_prog] = train_loss
                with torch.no_grad():
                    pred = self.model(self.x_valid)[:, 0]
                    loss_v = self.loss_fn(pred, self.y_valid.squeeze())
                    self.valid_loss_hist[epis // epis_prog] = loss_v.item() # / self.y_valid.size()[0]
                print(f"Episode: {epis:{3}} ({round(epis / num_epis * 100, 1)} %)", end = " | ")
                print(f"T. Loss: {self.train_loss_hist[epis // epis_prog]:{8}.{6}}", end = " | ")
                print(f"V. Loss: {self.valid_loss_hist[epis // epis_prog]:{8}.{6}}", end = " | ")
                print(f"ε-foc rad: {self.eps_f_rad:{5}.{4}}", end = " | ")
                print(f"ε ac-cent minmax: {np.round([min(self.eps_ac_centers), max(self.eps_ac_centers)], 1)}", end = " | ")
                print(f"Time: {round(time.time() - start_t)} s")

        print(f"Total Training Time: {round(time.time() - start_t)} s") # :{5}.{4}
        print("ε-focus action centers:\n", ' '.join(["{:.2f}".format(eac) for eac in self.eps_ac_centers]))
        return self.train_loss_hist, self.valid_loss_hist

    

    @classmethod
    def normalize_util_func(cls, u_func, minmax=[0, 1], num=101, verbose=False):
        st_arr = np.linspace(minmax[0], minmax[1], num, endpoint=True)
        util_arr = u_func(st_arr)
        uf_max, uf_min = max(util_arr), min(util_arr)
        diff, avg = uf_max - uf_min, (uf_max + uf_min) / 2.0
        normalized_uf = lambda x: (u_func(x) - avg) / diff
        if verbose: print(f"avg: {avg} | diff: {diff}")
        return normalized_uf

    @classmethod
    def get_determ_reward(cls, x_in, prob_arr, outcome_arr, util_func):
        y = np.zeros(len(x_in))
        p_arr, o_arr = np.array(prob_arr), np.array(outcome_arr)
        for idx, (s, a) in enumerate(x_in):
            # print((1-x) * np.ones(len(o_arr)) + x * o_arr)
            #                                    ↓ amount not risked ↓   +   ↓ amount risked ↓  -  ↓ util before bet ↓
            y[idx] = np.sum(p_arr * util_func(s * (1-a) * np.ones(len(o_arr)) + s * a * o_arr) - \
                            p_arr * util_func(s * np.ones(len(o_arr))))
        return y



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
    
    stat_mem_sz=(100, 100)
    lr = 1E-6 # 2E-5 1E-5
    st_range = np.array([0.0, 1.0])
    ac_range = np.array([0.0, 1.0])
    eps_foc_init = 1.5          # <=== epsilon focus implementation: initial radius of interval
    eps_foc_decay = 1 - 1E-3 # 1 - 1E-3
    eps_foc_min = 0.3   # min action radius
    eps_foc_gran = 0.1  # epsilon focus granularity, in terms of state values
    gamma = 0.0
    layers_sz = [15, 15] # [10, 10] [5, 10, 5] [12, 12] [15, 15]
    next_step_lookup = False    # True: q system | False: the simplest case, no looking up the next step (same as gamma=0)
    epochs_per_episode = 20      # number of cycles of training in self.learn_step per episode/call

    num_epis = 100_000 // epochs_per_episode
    epis_prog = 1_000 // epochs_per_episode

    # single-bet game
    env = betting_env.BettingEnvBinary(win_pr=prob_arr[1], loss_pr=prob_arr[0], win_fr=1.0, loss_fr=1.0, 
                                        start_cap=1, max_cap=st_minmax[1], min_cap=st_minmax[0], max_steps=1, 
                                        log_returns=False)

    agent = StatMemAgent(env=env, rng=rng, util_func=n_util_func, stat_mem_sz=stat_mem_sz, learn_rate=lr,
                        ac_range=ac_range, st_range=st_range, eps_foc_init=eps_foc_init, eps_foc_decay=eps_foc_decay, 
                        eps_foc_min=eps_foc_min, eps_foc_gran=eps_foc_gran, discount=gamma, layers_sz=layers_sz, 
                        next_step_lookup=next_step_lookup, epochs_per_episode=epochs_per_episode)
    

    # === ↓ Quick Tests Here ↓ ===


