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
        #                                    ↓ amount not risked ↓   +   ↓ amount risked ↓  -  ↓ util before bet ↓
        y[idx] = np.sum(p_arr * util_func(s * (1-a) * np.ones(len(o_arr)) + s * a * o_arr) - \
                        p_arr * util_func(s * np.ones(len(o_arr))))
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
                            st_minmax=[0.01, 2.0], epsilon=0.05, lr=0.002, num_epis=1000, epis_prog=100, stoch_mode=True): 
    nn.init.xavier_uniform_(model[0].weight)
    train_loss_hist = np.zeros(num_epis // epis_prog)
    valid_loss_hist = np.zeros(num_epis // epis_prog)
    # states_arr = np.zeros(num_epis + 1)     # extra space for s_n+1 after n steps
    states_arr = np.random.uniform(low=st_range[0], high=st_range[1], size=num_epis)    # initialize all states randomly
    actions_arr = np.zeros(num_epis)
    rewards_arr = np.zeros(num_epis)

    loss_fn = nn.MSELoss()  # MSELoss/L1Loss/SmoothL1Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not stoch_mode:  # Deterministic Training
        x_train, y_train = build_qnn_determ_set(prob_arr, outcome_arr, util_func=util_func,\
                                    st_range=st_range, ac_range=ac_range, num_st=20, num_ac=100)
        print("x_train:", x_train, "y_train:", y_train, "len y_train:", len(y_train))
    for epis in range(num_epis):
        if stoch_mode:
            state = states_arr[epis]
            action = select_action(model, state, ac_range=ac_range, epsilon=epsilon)
            next_st = get_stoch_next_state(state, action, prob_arr, outcome_arr)
            reward = get_state_change_reward(state, next_st, util_func=util_func)
            pred = model(torch.tensor([state, action], dtype=torch.float32))[0]
        else:
            state, action = x_train[epis % len(y_train)]  
            reward = y_train[epis % len(y_train)]
            pred = model(torch.tensor([state, action], dtype=torch.float32))[0]

        actions_arr[epis] = action
        rewards_arr[epis] = reward
        loss_tr = loss_fn(pred, torch.tensor(reward, dtype=torch.float32))
        loss_tr.backward()
        optimizer.step()
        optimizer.zero_grad()
            # if st_minmax[0] < next_st < st_minmax[1]:   # check for termination. For multi-trade
            #     states_arr[epis + 1] = next_st
            # else:
                # states_arr[epis + 1] = np.random.uniform(low=st_range[0], high=st_range[1])    # initialize state

        if epis % epis_prog == 0:
            train_loss_hist[epis // epis_prog] += loss_tr.item()
            with torch.no_grad():
                pred = model(x_valid)[:, 0]
                loss_v = loss_fn(pred, y_valid.squeeze())
                valid_loss_hist[epis // epis_prog] += loss_v.item() / y_valid.size()[0]
            print(f"Episode: {epis} | Training Loss: {train_loss_hist[epis // epis_prog]} | Validation Loss: {valid_loss_hist[epis // epis_prog]}")

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

def plot_history(train_loss_hist, valid_loss_hist, epsilon=0.00):
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(train_loss_hist)) + 1, train_loss_hist)
    plt.plot(np.arange(len(valid_loss_hist)) + 1, valid_loss_hist)
    plt.title(f"Q-NN Performance History\nEpsilon: {epsilon}")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.show()

def plot_performance(model, prob_arr, outcome_arr, num_st=10, num_ac=100, lr=1E-4, num_epis=100):
    x_valid, y_valid = build_qnn_determ_set(prob_arr, outcome_arr, util_func, st_range=[0.1, 1.0], ac_range=[0, 1.0], num_st=num_st, num_ac=num_ac)

    with torch.no_grad():
        y_pred = model(x_valid)[:, 0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    print(x_valid.shape, y_valid.shape, y_pred.shape)   # torch.Size([1000, 2]) torch.Size([1000]) torch.Size([1000])
    x_valid = x_valid.detach().numpy()
    y_valid = y_valid.detach().numpy()
    # plt.plot(x_valid[-num_ac:, 1], y_valid[-num_ac:])
    for s_idx in range(num_st):
        idx = s_idx * num_ac
        plt.plot(x_valid[idx:idx + num_ac, 1], y_valid[idx:idx + num_ac])
    plt.title("Validation Data")
    plt.xlabel("Investment Fraction")
    # plt.ylim((-.5, 0.2))
    plt.legend(x_valid[0:num_st * num_ac: num_ac, 0])   # labels/legends for different s value

    # print(f"x_valid: \n{x_valid} \n y_pred: \n{y_pred}")

    plt.subplot(1, 2, 2)
    y_pred  = y_pred.detach().numpy()
    for s_idx in range(num_st):
        idx = s_idx * num_ac
        plt.plot(x_valid[idx:idx + num_ac, 1], y_pred[idx:idx + num_ac])    # fix bug here
        # plt.plot(x_valid[0:num_ac, 1], y_pred[idx:idx + num_ac])
    plt.title(f"Model Performance\nLearning Rate: {lr}")
    plt.xlabel("Input X to Model: Investment Fraction")
    # plt.ylabel("Logarithmic Utility Function")
    # plt.legend(["Model", "Theory"])
    # plt.ylim((-2, 1))
    print(x_valid[0:num_st * num_ac:num_ac, 0].squeeze())
    plt.legend(x_valid[0:num_st * num_ac: num_ac, 0])   # labels/legends for different s values
    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    prob_arr = np.array([0.3, 0.7]) # [0.4, 0.6]
    outcome_arr = np.array([0.0, 2.0])
    st_range = np.array([0.01, 1.0])
    ac_range = np.array([0, 0.99])
    st_minmax = np.array([0.01, 2.0])
    util_func = lambda x: log_util(x, x_reg=1E-5)
    epsilon = 0.9

    stoch_mode = False  # True: Probabilistic Training | False: Deterministic Training
    if stoch_mode:
        lr = 2E-7
        num_epis = 500_000
        epis_prog = 50_000
    else:
        lr = 2E-5
        num_epis = 400_000
        epis_prog = 20_000
 
    model = build_multi_hidden_qnn(num_inputs=2, num_outputs=1, hid_size=[30, 30])   # print(model)

    x_valid, y_valid = build_qnn_determ_set(prob_arr, outcome_arr, util_func=util_func, \
                                    st_range=st_range, ac_range=ac_range, num_st=20, num_ac=20)
    
    train_loss_hist, valid_loss_hist = \
    train_qnn(model, x_valid, y_valid, prob_arr, outcome_arr, util_func=util_func, st_range=st_range, ac_range=ac_range,\
                st_minmax=st_minmax, epsilon=epsilon, lr=lr, num_epis=num_epis, epis_prog=epis_prog, stoch_mode=stoch_mode)

    plot_history(train_loss_hist, valid_loss_hist, epsilon=epsilon)

    plot_performance(model, prob_arr, outcome_arr, num_st=5, num_ac=21, lr=lr, num_epis=num_epis)


    # for i in range(20):   # test action selection
    #     action = select_action(model, 0.5, ac_range=[0.0, 1.0], epsilon=0.5, ac_granul=101)
    #     print(action)

    # # Manual Calculation:
    # a_in = np.linspace(0, 1, 21, endpoint=True)
    # # y_1 = 0.4 * np.log(1 - a_in) + 0.6 * np.log(1 + a_in)
    # y_2 = 0.4 * log_util(1 - a_in, x_reg=1E-10) + 0.6 * log_util(1 + a_in, x_reg=1E-10)
    # plt.figure()
    # # plt.plot(a_in, y_1)
    # plt.plot(a_in, y_2)
    # plt.ylim((-0.3, 0.1))
    # plt.show()


    # plt.figure()
    # for s in [0.1, 0.5, 1.0]:
    #     x_in = np.vstack(([s] * 21 , np.linspace(0, 1, 21, endpoint=True))).T # list of (s, a) for get_determ_reward
    #     # print(x_in)
    #     y = get_determ_reward_y(x_in, prob_arr, outcome_arr, util_func) # bug here
    #     plt.plot(x_in[:, 1], y)
    # # plt.ylim((-0.05, 0.05))
    # plt.show()
    
    # for i in range(20):
    #     next_st = np.random.uniform()
    #     reward = get_state_change_reward(0.5, next_st, util_func=util_func)
    #     print(next_st, reward, np.log(next_st/0.5))

    # st_arr = np.zeros(100)
    # for i in range(100):
    #     next_st = get_stoch_next_state(state=1.0, action=0.5, prob_arr=prob_arr, outcome_arr=outcome_arr)
    #     st_arr[i] = next_st
    # print(st_arr.sum())

