# Determining the utility function based on the probability of outcomes:
# Instead of supervised learning regression where theh value of the utility function is determined,
# we estimate the utility for an investment/betting fraction based on the frequency of different outcomes.
# Instead of preparing training data beforehand, the stochastic outcomes are generated in each epoch
# As inputs we have arrays of probabilities and outcomes, as well as the investment fractions
# For outputs, we have the average utility estimated
# 
# stochastic y generation using probability and outcome arrays:
# prob_arr:         [p1, p2, ... , pn]  with Σpi = 1
# outcome_arr:      [g1, g2, ... , gn]  with gi indicating the winning (or losing) fraction per amount betted. [0, Inf)
# gi are multiplicative: 
#   e.g. for a fair coin toss of double-or-nothing bet:
#   pi = [0.5, 0.5],    gi= [2.0, 0.0]

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def build_single_hidden_nn(num_inputs=1, num_outputs=1, hid_size=20):
    model = nn.Sequential(
                nn.Linear(num_inputs, hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, num_outputs),
            )
    return model

def build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[10, 10]):
    layer_list = []
    layer_sizes = zip([num_inputs] + list(hid_size), list(hid_size) + [num_outputs])

    for (num_in, num_out) in layer_sizes:
        layer_list.append(nn.Linear(num_in, num_out))
        layer_list.append(nn.ReLU())
    
    model = nn.Sequential(*layer_list)
    return model

def build_nn_stoch_train_set(prob_arr, outcome_arr, num_tr=100, start_pt=0.1, batch_size=1):
    x_train = np.linspace(start=start_pt, stop=10.0, num=num_tr, endpoint=True).reshape(num_tr, 1)
    y_train = torch.tensor(get_stoch_outcome_y(x_train, prob_arr, outcome_arr))
    x_train = torch.tensor(x_train, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dl

def get_stoch_outcome_y(x_in, prob_arr, outcome_arr):
    # x_train is the investment fraction
    # convert types
    p_arr, o_arr = np.array(prob_arr), np.array(outcome_arr)
    y_train = np.zeros(len(x_in))
    # check inputs
    if p_arr.shape != o_arr.shape: raise Exception(f"Input shapes not compatible: {p_arr.shape} vs {o_arr.shape}")
    # generate uniform randoms:
    rand_arr = np.random.uniform(low=0.0, high=1.0, size=len(x_in))
    print(rand_arr, rand_arr.sum())
    # cumulative probabilities
    p_cum = np.cumsum(p_arr)
    # print(p_cum)
    # outcome indices:
    for idx, x in enumerate(x_in):
        p_idx = np.nonzero(p_cum > rand_arr[idx])[0][0]
        # investment outcome based on investment fraction and the stochastic probabilities
        y_train[idx] = (1 - x) + x * o_arr[p_idx]
    print(y_train)
    # convert to log-utility
    y_train = get_log_util(y_train, x_reg=1E-10)
    return y_train

def build_nn_valid_set(num_val=35, start_pt=0.1):
    x_valid = np.linspace(start=start_pt, stop=10.0, num=num_val, endpoint=True).reshape(num_val, 1)
    y_valid = torch.tensor(get_log_util(x_valid, x_reg=1E-100), dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    return x_valid, y_valid

def get_log_util(x, x_reg=1E-10):
    y = np.log(np.maximum(x, x_reg))
    return y

def train_nn_probabilistic(model, x_valid, y_valid, num_tr=100, batch_size=5, lr=0.002, num_epochs=100):

    nn.init.xavier_uniform_(model[0].weight)

    train_loss_hist = np.zeros(num_epochs)
    valid_loss_hist = np.zeros(num_epochs)

    loss_fn = nn.L1Loss()  # L1 dist for probabilistic training
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    # model = build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[10, 10])

    # testing stochastic outcome: 50-50 Double or Nothing log utility, 10 times.
    y = get_stoch_outcome_y(np.ones(5), [0.5, 0.5], [2.0, 0.0])
    print(y)

    # print(model)