# Training a q-learning neural network for a single trade utility
# Unlike environments with finite number of actions in each step, both the state (capital or wealth)
# and the action (investment/betting fraction) are treated as continuous variables in each step 
# 
# Network Architecture:
# Input Nodes: The state (capital), the betting fraction
# Outputs Node: The q value q(s, a)
# In a single trade case, the expected utility of the outcome determines the q value
# This is a parametric form of finite-action q-networks, where one can afford a separate exit node
# for each action.
# 
# In the simplest case, we have single trade binary outcomes where neither the outcome gain/loss fraction 
# nor the respective probabilities depend on neither the captial nor the batting fraction.
# This is not the case in real investments, where the cost of borrowing will depend on the capital,
# and entering/exiting of larger positions can affect the market dynamics.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def build_multi_hidden_qnn(num_inputs=2, num_outputs=1, hid_size=[10, 10]):
    layer_list = []
    layer_sizes = zip([num_inputs] + list(hid_size), list(hid_size) + [num_outputs])

    for (num_in, num_out) in layer_sizes:
        layer_list.append(nn.Linear(num_in, num_out))
        layer_list.append(nn.ReLU())
    layer_list.pop()    # get rid of the last relu

    model = nn.Sequential(*layer_list)
    return model

def build_qnn_determ_set(prob_arr, outcome_arr, util_func, st_range=[0.0, 1.0], ac_range=[0, 0.99],\
                            num_st=20, num_ac=20):
    st_arr = np.linspace(start=st_range[0], stop=st_range[1], num=num_st, endpoint=True)
    ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True)

    ss, aa = np.meshgrid(st_arr, ac_arr, indexing='ij')

    x_valid = np.array(list(zip(ss.reshape(num_st * num_ac,), aa.reshape(num_st * num_ac,))))

    y_valid = get_determ_reward_y(x_valid, prob_arr, outcome_arr, util_func)
    # print(x_valid, y_valid)
    return torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)

def get_determ_reward_y(x_in, prob_arr, outcome_arr, util_func):
    y = np.zeros(len(x_in))
    p_arr, o_arr = np.array(prob_arr), np.array(outcome_arr)
    for idx, (s, a) in enumerate(x_in):
        # print((1-x) * np.ones(len(o_arr)) + x * o_arr)
        #                                    ↓ amount not risked ↓   +   ↓ amount risked ↓
        y[idx] = np.sum(p_arr * util_func(s * (1-a) * np.ones(len(o_arr)) + s * a * o_arr))
    return y

def log_util(x, x_reg=None, y_reg=None):
    if x_reg is not None:
        y = np.log(np.maximum(x, x_reg))
    elif y_reg is not None:
        y = np.maximum(np.log(x), y_reg)
    else:
        y = np.log(x)
    return y

def get_stoch_next_state(state, action, prob_arr, outcome_arr):
    p_arr, o_arr = np.array(prob_arr), np.array(outcome_arr)    # convert types
    if p_arr.shape != o_arr.shape: raise Exception(f"Input shapes not compatible: {p_arr.shape} vs {o_arr.shape}")
    # cumulative probabilities
    p_cum = np.cumsum(p_arr)
    r = np.random.uniform(low=0.0, high=1.0)
    # outcome index based on random r and cumul probabs
    idx = np.nonzero(p_cum > r)[0][0]       # print(idx)
    # next state based on: original state, action (risked cap) and stoch outcome:
    #        ↓ amount not risked ↓  +   ↓ amount risked * outcome ↓
    next_st =   (state - action)    +   action * o_arr[idx]
    return next_st

def get_state_change_reward(state, next_st, util_func):
    reward = util_func(next_st) - util_func(state)
    return reward

def train_qnn(model, x_valid, y_valid, prob_arr, outcome_arr, util_func, st_range=[0.0, 1.0], ac_range=[0, 0.99],\
                            epsilon=0.05, lr=0.002, num_episodes=1000, epis_prog=100, stoch_mode=True): 
    nn.init.xavier_uniform_(model[0].weight)
    train_loss_hist = np.zeros(num_episodes)
    valid_loss_hist = np.zeros(num_episodes)
    states_arr = np.zeros(num_episodes + 1)
    actions_arr = np.zeros(num_episodes)
    rewards_arr = np.zeros(num_episodes)
    states_arr[0] = np.random.uniform(low=st_range[0], high=st_range[1])    # initialize state

    loss_fn = nn.MSELoss()  # MSELoss/L1Loss/SmoothL1Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not stoch_mode:  # Deterministic Training
        x_train, y_train = build_qnn_determ_set(prob_arr, outcome_arr, util_func=util_func,\
                                st_range=st_range, ac_range=ac_range, num_st=20, num_ac=20)
        # train_dataset = TensorDataset(x_train, y_train)
        # train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for episode in range(num_episodes):
        if stoch_mode:
            state = states_arr[episode]
            action = select_action(model, state, ac_range=ac_range, epsilon=epsilon)
            pred = model(torch.tensor([state, action], dtype=torch.float32))[:, 0]
            next_st = get_stoch_next_state(state, action, prob_arr, outcome_arr)
            reward = get_state_change_reward(state, next_st, util_func=util_func)
            states_arr[episode + 1] = next_st
            rewards_arr[episode] = reward
            loss_tr = loss_fn(pred, torch.tensor(reward, dtype=torch.float32))
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()




    return train_loss_hist, valid_loss_hist

def select_action(model, state, ac_range=[0.0, 1.0], epsilon=0.05, ac_granul=101):
    # ac_granul: granularity of the action space for q determination
    s_arr = np.array([state] * ac_granul)
    a_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=ac_granul, endpoint=True)
    sa_arr = np.vstack((s_arr, a_arr)).transpose()  # rows: ac_granul, columns: 2   print(sa_arr.shape)
    sa_tensor = torch.tensor(sa_arr, dtype=torch.float32)
    # q_arr = np.zeros(ac_granul)
    r = np.random.uniform(low=0, high=1)
    if r < epsilon:
        action = np.random.uniform(low=ac_range[0], high=ac_range[1])
    else:
        with torch.no_grad():
            q_arr = model(sa_tensor).detach().numpy()[:]
            idx = np.argmax(q_arr)
            action=  a_arr[idx]
    return action


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    prob_arr = [0.4, 0.6]
    outcome_arr = [0.0, 2.0]
    stoch_mode = True  # True: Probabilistic Training | False: Deterministic Training
    util_func = lambda x: log_util(x, y_reg=-20)

    model = build_multi_hidden_qnn(num_inputs=2, num_outputs=1, hid_size=[20, 30])   # print(model)

    x_valid, y_valid = build_qnn_determ_set(prob_arr, outcome_arr, util_func=util_func, \
                                st_range=[0.0, 1.0], ac_range=[0, 1], num_st=20, num_ac=20)

    # for i in range(20):   # test action selection
    #     action = select_action(model, 0.5, ac_range=[0.0, 1.0], epsilon=0.5, ac_granul=101)
    #     print(action)

    # for i in range(20):
    #     next_st = np.random.uniform()
    #     reward = get_state_change_reward(0.5, next_st, util_func=util_func)
    #     print(next_st, reward, np.log(next_st/0.5))

    # st_arr = np.zeros(100)
    # for i in range(100):
    #     next_st = get_stoch_next_state(state=1.0, action=0.5, prob_arr=prob_arr, outcome_arr=outcome_arr)
    #     st_arr[i] = next_st
    # print(st_arr.sum())

