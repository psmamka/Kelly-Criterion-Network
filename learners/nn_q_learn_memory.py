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

from nn_single_trade_q_learn import \
    log_util, get_determ_reward_y, get_stoch_next_state, get_state_change_reward

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class QInvestAgent:
    def __init__(self, env, util_func=lambda x: x, rng=np.random.default_rng(), mem_size=int(1E6), recall_size=None,
                    learn_rate=1E-3, eps_init=1.0, eps_decay=0.999, eps_min=0, discount=1.0, layers_sz=[20, 20]):
        
        self.env = env
        self.util_func = util_func
        
        self.rng = rng

        self.in_sz = 2  # state (capital), action (invest ratio)
        self.out_sz = 1 # Expected utility for a given (s, a) pair

        self.memory = deque(maxlen=mem_size)
        self.recall_sz = recall_size if recall_size is not None else mem_size
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
    
    def recall(self, recall_mech=None):
        if recall_mech is None: recall_mech = 'random'      # 'random' | 'recent' | 'weighted' | 'smart' 

        samples = self.rng.choice(self.memory, size=self.recall_sz, replace=True)
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
    
    def update_eps(self):
        if self.eps > self.eps_min: self.eps *= self.eps_decay
    
    def state_change_reward(self, state, next_st):
        return self.util_func(next_st) - self.util_func(state)    # utility reward

    def learn_step(self, recall_samples):
        recall_inputs, recall_targets = [], []
        # print(recall_inputs.shape)

        for idx, transition in enumerate(recall_samples):
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
    lr = 1E-3
    eps_init = 1.0
    eps_decay = 1 - 1E-3
    eps_min = 0
    gamma = 1.0
    layers_sz = [30, 30]

    # single bet game
    env = betting_env.BettingEnvBinary(win_pr=prob_arr[1], loss_pr=prob_arr[0], win_fr=1.0, loss_fr=1.0, 
                                        start_cap=1, max_cap=st_minmax[1], min_cap=st_minmax[0], max_steps=1, log_returns=False)

    agent = QInvestAgent(env=env, rng=rng, util_func=util_func, mem_size=mem_size, learn_rate=lr, eps_init=eps_init, 
                            eps_decay=eps_decay, eps_min=eps_min, discount=gamma, layers_sz=layers_sz)

    # quick test of env and memory
    # for i in range(10):
    #     env.reset()
    #     s = env.cur_cap
    #     next_st, r, term = env.step(bet_size=0.5)
    #     print(f"state: {s}, next_st: {next_st}, r: {r}, term: {term}")
    # print(agent.memory[1000])

    # test of recall and learn_Step
    samples = agent.recall()    # print(samples)
    loss = agent.learn_step(samples)
    print(loss)


