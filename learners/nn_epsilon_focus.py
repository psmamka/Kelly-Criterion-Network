# Continuation of nn_memory. Implementing the epsilon focus mechanism

# Epsilon-Focus Implementation
#   Rational: We are primarily interested in finding the max utility point
#       - Exact values of the utility function at other points are not crucial
#       - only insofar as they can help find the max point. e.g. discard the sharp tail for log util at a=1
#   This is roughly a continuous equivalent to the discrete epsilon-greedy method
#   The idea is to progressively focus sampling to areas where we have higher utility
#   One complication is that different states have different optimal actions
#   - we can still apply epsilon focus, the center interval is going to be different
#   - we have to keep track of optmal actions
#   In one simple implementation:
#       - initially we select uniform samples from an interval that covers the entire action space
#       - In each step, for a given state, estimate the optimal action. 
#           -- center the epsilon-focus interval on the estimate of max util 
#       - As the network is trained the the focus interval is centered at the max util point,
#           - different max util, hence different focus center for different states
#       - The size/diameter of the action focus is gradually reduced (up to some minimum)
#       - with further training, the center and diameter are adjusted (state dependent)

import sys
try:    # for intellisense
    from .. envs import betting_env
except:
    sys.path.append('..')
    from envs import betting_env

import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

from nn_single_trade_q_learn import log_util

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class QInvestAgent:
    def __init__(self, env, util_func=lambda x: x, rng=np.random.default_rng(), mem_size=int(1E4), recall_size=None,
                    recall_mech='recent', learn_rate=1E-3, ac_range=[0.0, 1.0], st_range=[0.0, 1.0],  eps_foc_init=1.0, 
                    eps_foc_decay=0.999, eps_foc_min=0.5, eps_foc_gran=0.1, discount=1.0, layers_sz=[20, 20], 
                    next_step_lookup=True, epochs_per_episode=1):
        
        self.env = env
        self.util_func = util_func
        
        self.rng = rng

        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair

        self.memory = deque(maxlen=mem_size)
        self.recall_sz = recall_size if recall_size is not None else mem_size
        self.recall_mech = recall_mech        # recall mechanisms: 'recent' | 'random' | 'weighted' | 'smart'
        self.lr = learn_rate

        self.ac_range = ac_range
        self.st_range = st_range

        self.eps_f = eps_foc_init         # epsilon greedy parameters
        self.eps_f_dec = eps_foc_decay
        self.eps_f_min = eps_foc_min
        self.eps_f_gran = eps_foc_gran
        self._epsilon_focus_init()

        self.gamma = discount
        self.next_lookup = next_step_lookup   # q: predictions of next step used for current step optimization
        self.epoc_per_epis = epochs_per_episode

        self._build_qnn(in_sz=self.in_sz, out_sz=self.out_sz, layers_sz=layers_sz)
        self._mem_init()
        # self._focus_mem_init()      # <=== keeping track of the focus centers. Then need to update in each epoch.
    
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
    
    def _mem_init(self):
        for _ in range(self.memory.maxlen):
            s = self.rng.uniform()
            a = self.rng.uniform()

            self.env.reset(start_cap=s)
            next_s, _, _ = self.env.step(bet_size= s * a)
            r = self.state_change_reward(state=s, next_st=next_s)
            tr = Transition(state=s, action=a, reward=r, next_state=next_s)
            self.remember(tr)
    
    def _epsilon_focus_init(self, verbose=False):
        # number of epsilon centers to keep track of: (st_max - st_min) / epsilon_granularity + 1 
        # (+1 because exclusive of boundaries)
        num_eps_states = int((self.st_range[1] - self.st_range[0]) / self.eps_f_gran + 1)
        self.eps_f_states = np.linspace(self.st_range[0], self.st_range[1], num_eps_states, endpoint= True)
        avg_action = (self.ac_range[1] - self.ac_range[0]) / 2.0    # middle point for possible actions used for init
        self.eps_f_centers = np.ones(num_eps_states) * avg_action
        if verbose:
            print(f"eps_f granular states: {self.eps_f_states} \neps_f action centers: {self.eps_f_centers}")

    def remember(self, tr):
        self.memory.append(tr)
    
    def recall(self):
        mem_list = list(self.memory)
        if self.recall_mech == 'recent':
            samples = mem_list[-self.recall_sz:]
        elif self.recall_mech == 'random':  
            samples = self.rng.choice(mem_list, size=self.recall_sz, replace=False)
        return samples
    
    def select_action(self, state, ac_range=[0.0, 1.0], ac_granul=21):
        # ac_granul: granularity of the action space for q determination/action selection
        r = self.rng.uniform(low=0, high=1)     # <=== update for focus here
        if r < self.eps_f:    # random action
            action = self.rng.uniform(low=ac_range[0], high=ac_range[1])
        else:
            s_arr = np.array([state] * ac_granul)
            a_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=ac_granul, endpoint=True)

            sa_arr = np.vstack((s_arr, a_arr)).transpose()  # rows: ac_granul, columns: 2   print(sa_arr.shape)
            sa_tensor = torch.tensor(sa_arr, dtype=torch.float32)
            with torch.no_grad():
                q_arr = self.model(sa_tensor).detach().numpy()[:]
                idx = np.argmax(q_arr)
                action = a_arr[idx]
        return action
    
    def _update_eps_f(self):    # update epsilon focus
        if self.eps_f > self.eps_f_min: self.eps_f *= self.eps_f_dec
    
    def state_change_reward(self, state, next_st):
        return self.util_func(next_st) - self.util_func(state)    # utility reward

    def learn_step(self, recall_samples):
        # sz = recall_samples.shape[0]    # print(f"sample size: {sz}")
        recall_inputs, recall_targets = [], []
        # print(recall_inputs.shape)

        for transition in recall_samples:
            s, a, r, next_s = transition

            if self.next_lookup: # the q method: use predictions for the next steps
                with torch.no_grad():
                    next_a = self.select_action(next_s)
                    next_pred = self.model(torch.tensor([next_s, next_a], dtype=torch.float32))[0]
                    target = torch.tensor(r, dtype=torch.float32) + self.gamma * next_pred  # tensor
            else:
                target = torch.tensor(r, dtype=torch.float32)   # no next step for now
            
            recall_inputs.append([s, a])
            recall_targets.append(target)

        # train_dataset = TensorDataset(np.array(recall_inputs).reshape((sz, 2)), recall_targets.reshape((sz, 1)))
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
    
    def train_qnn(self, num_epis=int(1E5), epis_prog=int(1E3)):
        start_t = time.time()
        nn.init.xavier_uniform_(self.model[0].weight)
        # for layer in self.model:    # initialize linear layers
        #     if type(layer) == nn.Linear: nn.init.xavier_uniform_(layer.weight)

        self.train_loss_hist = np.zeros(num_epis // epis_prog)
        self.valid_loss_hist = np.zeros(num_epis // epis_prog)

        if self.x_valid is None or self.y_valid is None:
            raise Exception("validation data need to be initialized")

        for epis in range(num_epis):
            # single episode logic
            s = self.rng.uniform()
            
            self.env.reset(start_cap = s)
            a = self.select_action(state = s, ac_range=[0, 1], ac_granul=21)
            next_s, _, _ = env.step(bet_size= s * a)
            r = self.state_change_reward(state=s, next_st=next_s)
            tr = Transition(state=s, action=a, reward=r, next_state=next_s)
            self.remember(tr)
            recall_samples = self.rng.permutation(self.recall())
            train_loss = self.learn_step(recall_samples=recall_samples)
            self._update_eps_f()

            if epis % epis_prog == 0:   # progress report logic
                self.train_loss_hist[epis // epis_prog] = train_loss
                with torch.no_grad():
                    pred = self.model(self.x_valid)[:, 0]
                    loss_v = self.loss_fn(pred, self.y_valid.squeeze())
                    self.valid_loss_hist[epis // epis_prog] = loss_v.item() # / self.y_valid.size()[0]
                print(f"Episode: {epis:{3}}", end = " | ")
                print(f"Training Loss: {self.train_loss_hist[epis // epis_prog]:{9}.{6}}", end = " | ")
                print(f"Validation Loss: {self.valid_loss_hist[epis // epis_prog]:{9}.{6}}", end = " | ")
                print(f"Time: {time.time() - start_t:{5}.{4}} s")

        print(f"Total Training Time: {time.time() - start_t:{5}.{4}} s")
        return self.train_loss_hist, self.valid_loss_hist

    def generate_validation_data(self, prob_arr, outcome_arr, util_func, st_range=[0.0, 1.0],  ac_range=[0, 0.99], num_st=20, num_ac=20):
        st_arr = np.linspace(start=st_range[0], stop=st_range[1], num=num_st, endpoint=True)
        ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True)

        ss, aa = np.meshgrid(st_arr, ac_arr, indexing='ij')

        x_valid = np.array(list(zip(ss.reshape(num_st * num_ac,), aa.reshape(num_st * num_ac,))))
        y_valid = self.__class__.get_determ_reward(x_valid, prob_arr, outcome_arr, util_func)
        # self.x_valid, self.y_valid = torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)
        self.x_valid, self.y_valid = map(lambda u: torch.tensor(u, dtype=torch.float32), [x_valid, y_valid])
        self.num_st_val, self.num_ac_val = num_st, num_ac
        return self.x_valid, self.y_valid

    def plot_training_history(self, epis_prog=1000):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(self.train_loss_hist)) * epis_prog + 1, self.train_loss_hist)
        plt.title(f"Q-NN Performance History: Training\nMemory: {self.memory.maxlen}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2) 
        plt.plot(np.arange(len(self.valid_loss_hist)) * epis_prog + 1, self.valid_loss_hist)
        plt.title(f"Q-NN Performance History: Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def plot_performance(self, show_legends=True, num_st=11, num_ac=11):
        if self.x_valid is None or self.y_valid is None:
            raise Exception("validation data need to be initialized")
        
        with torch.no_grad():
            y_pred = self.model(self.x_valid)[:, 0]

        plt.figure(figsize=(10, 5))

        ax = plt.subplot(1, 2, 1)    # validation data plot
        print(self.x_valid.shape, self.y_valid.shape, y_pred.shape)   # torch.Size([400, 2]) torch.Size([400]) torch.Size([400])
        x_valid = self.x_valid.detach().numpy()
        y_valid = self.y_valid.detach().numpy()
        # plt.plot(x_valid[-num_ac:, 1], y_valid[-num_ac:])
        # print(x_valid[0:self.num_st_val * self.num_ac_val:self.num_ac_val, 0].squeeze())

        legends = [f"s = {z:{3}.{2}}" for z in x_valid[0:self.num_st_val * self.num_ac_val: self.num_ac_val, 0]]
        # print(legends)

        for s_idx in range(self.num_st_val):
            idx = s_idx * self.num_ac_val
            plt.plot(x_valid[idx:idx + self.num_ac_val, 1], y_valid[idx:idx + self.num_ac_val], label=legends[s_idx])
        plt.title("Validation Data")
        plt.xlabel("Investment Fraction")
        handles, _ = ax.get_legend_handles_labels()
        if show_legends: ax.legend([handles[0], handles[-1]], [legends[0], legends[-1]])
        # if show_legends:
        #     plt.legend(x_valid[0:self.num_st_val * self.num_ac_val: self.num_ac_val, 0])   # labels/legends for different s value

        ax2 = plt.subplot(1, 2, 2)    # performance data plot
        y_pred  = y_pred.detach().numpy()
        for s_idx in range(self.num_st_val):
            idx = s_idx * self.num_ac_val
            plt.plot(x_valid[idx:idx + self.num_ac_val, 1], y_pred[idx:idx + self.num_ac_val], label=legends[s_idx])
            # plt.plot(x_valid[0:num_ac, 1], y_pred[idx:idx + num_ac])
        plt.title(f"Model Performance\nLearning Rate: {lr}")
        plt.xlabel("Investment Fraction")
        
        handles, _ = ax2.get_legend_handles_labels()
        if show_legends: ax2.legend([handles[0], handles[-1]], [legends[0], legends[-1]])
        plt.show()

        y_2d_test =  y_valid.reshape((num_st, num_ac))
        y_2d_pred =  y_pred.reshape((num_st, num_ac))
        num_levs = 25
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(y_2d_test, levels=num_levs)
        plt.colorbar()
        plt.title(f"Target Contour")
        plt.xlabel("Action: Investment Fraction")
        plt.ylabel("State: Capital")
        plt.xticks(ticks=np.arange(num_ac), labels=[f"{z:{4}.{2}}" for z in np.linspace(0, 1, num=num_ac, endpoint=True)])
        plt.yticks(ticks=np.arange(num_st), labels=[f"{z:{4}.{2}}" for z in np.linspace(0, 1, num=num_st, endpoint=True)])
        # plt.legend(["Model", "Theory"])
        plt.subplot(1, 2, 2)
        plt.contourf(y_2d_pred, levels=num_levs)
        plt.colorbar()
        plt.title(f"Predicted Contour")
        plt.xlabel("Action: Investment Fraction")
        plt.ylabel("State: Capital")
        plt.xticks(ticks=np.arange(num_ac), labels=[f"{z:{4}.{2}}" for z in np.linspace(0, 1, num=num_ac, endpoint=True)])
        plt.yticks(ticks=np.arange(num_st), labels=[f"{z:{4}.{2}}" for z in np.linspace(0, 1, num=num_st, endpoint=True)])

        plt.show()
    
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
    
if __name__ == '__main__':
    rng = np.random.default_rng()
    torch.manual_seed(1)

    # betting env properties
    prob_arr = np.array([0.3, 0.7]) #
    outcome_arr = np.array([0.0, 2.0])
    # validation
    num_st, num_ac = 10, 21
    # validation data grid
    st_range = np.array([0.0, 1.0])
    ac_range = np.array([0.0, 1.0])
    st_minmax = np.array([0.0, 2.0])
    util_func = lambda x: log_util(x, x_reg=1E-3)
    # util_func = lambda x: x
    normalize_uf = True     # True: Normalize utility function to [-1 +1] range | False: use raw util func
    if normalize_uf: 
        n_util_func = QInvestAgent.normalize_util_func(util_func, minmax=st_minmax, num=11, verbose=True)
    else:
        n_util_func = util_func
    mem_size = int(4E3)     # 4E3 recent, 1E4 for random
    recall_mech = 'recent' # 'recent' | 'random'
    recall_size = mem_size # // 2
    lr = 2E-5 # 1E-5
    eps_foc_init = 1.0          # <=== epsilon focus implementation: radius of interval
    eps_foc_decay = 1 # 1 - 1E-4
    eps_foc_min = 0.1
    eps_foc_gran = 0.1  # epsilon focus granularity, in terms of state values
    gamma = 0.0
    layers_sz = [10, 10]
    next_step_lookup = False    # True: q system | False: the simplest case, no looking up the next step (same as gamma=0)
    epochs_per_episode = 10      # number of cycles of training in self.learn_step per episode/call

    num_epis = 40_000 // epochs_per_episode
    epis_prog = 1000 // epochs_per_episode

    # single-bet game
    env = betting_env.BettingEnvBinary(win_pr=prob_arr[1], loss_pr=prob_arr[0], win_fr=1.0, loss_fr=1.0, 
                                        start_cap=1, max_cap=st_minmax[1], min_cap=st_minmax[0], max_steps=1, 
                                        log_returns=False)

    agent = QInvestAgent(env=env, rng=rng, util_func=n_util_func, mem_size=mem_size, recall_mech=recall_mech, recall_size=recall_size, 
                        learn_rate=lr, ac_range=ac_range, st_range=st_range, eps_foc_init=eps_foc_init, eps_foc_decay=eps_foc_decay, 
                        eps_foc_min=eps_foc_min, eps_foc_gran=eps_foc_gran, discount=gamma, layers_sz=layers_sz, 
                        next_step_lookup=next_step_lookup, 
                        epochs_per_episode=epochs_per_episode)

    agent.generate_validation_data(prob_arr, outcome_arr, util_func=n_util_func,
                                        st_range=st_range, ac_range=ac_range, num_st=num_st, num_ac=num_ac)
    
    agent.train_qnn(num_epis=num_epis, epis_prog=epis_prog)

    agent.plot_training_history(epis_prog=epis_prog)

    agent.plot_performance(show_legends=True, num_st=num_st, num_ac=num_ac)
    
    # # quick test of env and memory
    # for i in range(10):
    #     env.reset()
    #     s = env.cur_cap
    #     next_st, r, term = env.step(bet_size=0.5)
    #     print(f"state: {s}, next_st: {next_st}, r: {r}, term: {term}")
    # print(agent.memory[1000])

    # # quick test of recall and learn_Step
    # samples = agent.recall()    
    # print(samples)
    # loss = agent.learn_step(samples)
    # print(loss)

    # # quick test of layer weight init
    # agent.train_qnn()
    # print(agent.model)
    # print(agent.model[0].weight)

    # # test of get_determ_reward():
    # plt.figure()
    # s_arr = [0.1, 0.5, 1.0]
    # for s in s_arr:
    #     x_in = np.vstack(([s] * 21 , np.linspace(0, 1, 21, endpoint=True))).T # list of (s, a) for get_determ_reward
    #     y = QInvestAgent.get_determ_reward(x_in, prob_arr, outcome_arr, util_func)
    #     plt.plot(x_in[:, 1], y)
    # plt.legend(s_arr)
    # plt.show()

    # # test of generate_validation_data()
    # x_v, y_v = agent.generate_validation_data(prob_arr, outcome_arr, util_func=util_func,
    #                                 st_range=st_range, ac_range=ac_range, num_st=20, num_ac=20)
    # print(x_v.shape, y_v.shape) # torch.Size([400, 2]) torch.Size([400])
    # y = y_v.detach().numpy().reshape((20, 20))
    # print(y.shape)
    # plt.contourf(y)
    # plt.colorbar()
    # plt.xlabel('action: betting fraction')
    # plt.ylabel('state: capital')
    # plt.show()

    # # quick test of epsilon granularity
    # agent._epsilon_focus_init(verbose = True)
