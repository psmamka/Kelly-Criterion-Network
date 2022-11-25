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
    print(x_valid, y_valid)
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

def train_nn(model, x_valid, y_valid, prob_arr, outcome_arr, num_tr=100, batch_size=5, \
                            lr=0.002, num_epochs=100, epoch_prog=100, stoch_mode=True): 
    train_loss_hist = np.zeros(num_epochs)
    valid_loss_hist = np.zeros(num_epochs)

    return train_loss_hist, valid_loss_hist


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    prob_arr = [0.4, 0.6]
    outcome_arr = [0.0, 2.0]
    stoch_mode = True  # True: Probabilistic Training | False: Deterministic Training
    util_func = lambda x: log_util(x, y_reg=-10)

    # model = build_multi_hidden_qnn(num_inputs=2, num_outputs=1, hid_size=[20, 30])   # print(model)

    x_valid, y_valid = build_qnn_determ_set(prob_arr, outcome_arr, util_func=util_func, \
                                st_range=[0.0, 1.0], ac_range=[0, 1], num_st=20, num_ac=20)

