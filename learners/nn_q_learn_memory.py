# A neural network with a memroy queue (q-learning) for optimal investment ratio
# The general layout is roughly parallel to that of Raschka et al. Python Machine Learning, 
# PyTorch Edition 2022 Ch.19 DQN examples. Due to continuous action, the output layer in
# our network is just the q value, with multiple inputs (state/action) tried to find the
# q-optimal action.
# 
# 
# General Steps:
# - Initialize Neural Network
# - Initialize/fill the memory double-ended queue
# - Train the network for various initial states
#   -- develop single learn step, where samples from memory are used to train
#   -- in every training step:
#     --- reset env
#     --- choose action (epsilon-greedy, with decaying epsilon)
#     --- adjust epsilon
#     --- memorize the transition
#     --- take samples from the memory (replay/recall) and train
#     --- record get training loss
#     --- optional: calculate validation loss
#     --- progress logic
# - plot training history
# - plot performance, compare to theoretical util

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
import matplotlib.pyplot as plt

from nn_single_trade_q_learn import log_util

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class QInvestAgent:
    def __init__(self, env, util_func=lambda x: x, rng=np.random.default_rng(), mem_size=int(1E6), recall_size=None,
                    recall_mech='recent', learn_rate=1E-3, eps_init=1.0, eps_decay=0.999, eps_min=0, discount=1.0, 
                    layers_sz=[20, 20]):
        
        self.env = env
        self.util_func = util_func
        
        self.rng = rng

        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair

        self.memory = deque(maxlen=mem_size)
        self.recall_sz = recall_size if recall_size is not None else mem_size
        self.recall_mech = recall_mech        # recall mechanisms: 'recent' | 'random' | 'weighted' | 'smart'
        self.lr = learn_rate

        self.eps = eps_init         # epsilon greedy parameters
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = discount

        self._build_qnn(in_sz=self.in_sz, out_sz=self.out_sz, layers_sz=layers_sz)
        self._mem_init()
    
    def _build_qnn(self, in_sz=2, out_sz=1, layers_sz=[20, 20]):
        layer_list = []
        layer_sizes = zip([in_sz] + list(layers_sz), list(layers_sz) + [out_sz])

        for idx, (num_in, num_out) in enumerate(layer_sizes):
            layer_list.append(nn.Linear(num_in, num_out))
            if idx < len(layers_sz): layer_list.append(nn.ReLU())

        self.model = nn.Sequential(*layer_list) # print(self.model)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    
    def _mem_init(self):
        for i in range(self.memory.maxlen):
            s, a = rng.uniform(size=2)
            env.reset(start_cap=s)
            next_s, _, _ = env.step(bet_size= s * a)
            r = self.state_change_reward(state=s, next_st=next_s)
            tr = Transition(state=s, action=a, reward=r, next_state=next_s)
            self.remember(tr)
    
    def remember(self, tr):
        self.memory.append(tr)
    
    def recall(self):
        mem_list = list(self.memory)
        if self.recall_mech == 'recent':
            samples = mem_list[-self.recall_sz:]
        elif self.recall_mech == 'random':
            samples = self.rng.choice(mem_list, size=self.recall_sz, replace=True)
        return samples
    
    def select_action(self, state, ac_range=[0.0, 1.0], ac_granul=101):
        # ac_granul: granularity of the action space for q determination/action selection
        r = self.rng.uniform(low=0, high=1)
        if r < self.eps:
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
    
    def _update_eps(self):
        if self.eps > self.eps_min: self.eps *= self.eps_decay
    
    def state_change_reward(self, state, next_st):
        return self.util_func(next_st) - self.util_func(state)    # utility reward

    def learn_step(self, recall_samples):
        recall_inputs, recall_targets = [], []
        # print(recall_inputs.shape)

        for transition in recall_samples:
            s, a, r, next_s = transition

            with torch.no_grad():
                next_a = self.select_action(next_s)
                next_pred = self.model(torch.tensor([next_s, next_a], dtype=torch.float32))[0]
                target = r + self.gamma * next_pred # tensor
            
            recall_inputs.append([s, a])
            recall_targets.append(target)

        self.optimizer.zero_grad()
        recall_pred = self.model(torch.tensor(recall_inputs, dtype=torch.float32)).squeeze()  # [0]
        loss = self.loss_fn(recall_pred, torch.stack(recall_targets, dim=0))
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_qnn(self, x_valid, y_valid, num_epis=int(1E5), epis_prog=int(1E3)):
        for layer in self.model:    # initialize linear layers
            if type(layer) == nn.Linear: nn.init.xavier_uniform_(layer.weight)

        train_loss_hist = np.zeros(num_epis // epis_prog)
        valid_loss_hist = np.zeros(num_epis // epis_prog)

        for epis in range(num_epis):
            # single episode logic
            s = rng.uniform()
            env.reset(start_cap = s)
            a = self.select_action(state = s, ac_range=[0, 1], ac_granul=101)
            next_s, _, _ = env.step(bet_size= s * a)
            r = self.state_change_reward(state=s, next_st=next_s)
            tr = Transition(state=s, action=a, reward=r, next_state=next_s)
            self.remember(tr)
            recall_samples = self.recall()
            train_loss = self.learn_step(recall_samples=recall_samples)
            self._update_eps()

            if epis % epis_prog == 0:   # progress report logic
                train_loss_hist[epis // epis_prog] += train_loss
                with torch.no_grad():
                    pred = self.model(x_valid)[:, 0]
                    loss_v = self.loss_fn(pred, y_valid.squeeze())
                    valid_loss_hist[epis // epis_prog] += loss_v.item() / y_valid.size()[0]
                print(f"Episode: {epis} | Training Loss: {train_loss_hist[epis // epis_prog]} | Validation Loss: {valid_loss_hist[epis // epis_prog]}")
        return train_loss_hist, valid_loss_hist

    @classmethod
    def generate_validation_data(cls, prob_arr, outcome_arr, util_func, st_range=[0.0, 1.0],  ac_range=[0, 0.99], num_st=20, num_ac=20):

        st_arr = np.linspace(start=st_range[0], stop=st_range[1], num=num_st, endpoint=True)
        ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True)

        ss, aa = np.meshgrid(st_arr, ac_arr, indexing='ij')

        x_valid = np.array(list(zip(ss.reshape(num_st * num_ac,), aa.reshape(num_st * num_ac,))))
        y_valid = cls.get_determ_reward(x_valid, prob_arr, outcome_arr, util_func)
        return torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)
    
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

    prob_arr = np.array([0.3, 0.7]) #
    outcome_arr = np.array([0.0, 2.0])
    st_range = np.array([0.01, 1.0])
    ac_range = np.array([0, 0.99])
    st_minmax = np.array([0.01, 5.0])
    util_func = lambda x: log_util(x, x_reg=1E-5)
    # util_func = lambda x: x
    mem_size = int(1E4)
    recall_mech = 'recent'
    lr = 1E-3
    eps_init = 1.0
    eps_decay = 1 - 1E-3
    eps_min = 0
    gamma = 1.0
    layers_sz = [30, 30]
    num_epis = 100
    epis_prog = 10

    # single-bet game
    env = betting_env.BettingEnvBinary(win_pr=prob_arr[1], loss_pr=prob_arr[0], win_fr=1.0, loss_fr=1.0, 
                                        start_cap=1, max_cap=st_minmax[1], min_cap=st_minmax[0], max_steps=1, log_returns=False)

    agent = QInvestAgent(env=env, rng=rng, util_func=util_func, mem_size=mem_size, recall_mech=recall_mech, learn_rate=lr, 
                        eps_init=eps_init, eps_decay=eps_decay, eps_min=eps_min, discount=gamma, layers_sz=layers_sz)

    x_valid, y_valid = QInvestAgent.generate_validation_data(prob_arr, outcome_arr, util_func=util_func,
                                        st_range=st_range, ac_range=ac_range, num_st=20, num_ac=20)
    
    agent.train_qnn(x_valid, y_valid, num_epis=num_epis, epis_prog=epis_prog)

    # # quick test of env and memory
    # for i in range(10):
    #     env.reset()
    #     s = env.cur_cap
    #     next_st, r, term = env.step(bet_size=0.5)
    #     print(f"state: {s}, next_st: {next_st}, r: {r}, term: {term}")
    # print(agent.memory[1000])

    # # test of recall and learn_Step
    # samples = agent.recall()    # print(samples)
    # loss = agent.learn_step(samples)
    # print(loss)

    # # test of layer weight init
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
    # x_v, y_v = QInvestAgent.generate_validation_data(prob_arr, outcome_arr, util_func=util_func,
    #                                 st_range=st_range, ac_range=ac_range, num_st=20, num_ac=20)
    # print(x_v.shape, y_v.shape) # torch.Size([400, 2]) torch.Size([400])
    # y = y_v.detach().numpy().reshape((20, 20))
    # print(y.shape)
    # plt.imshow(y, cmap='gray')
    # plt.colorbar()
    # plt.xlabel('action: betting fraction')
    # plt.ylabel('state: capital')
    # plt.show()



