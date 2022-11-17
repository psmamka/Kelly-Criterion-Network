# Determining the utility function based on the probability of outcomes:
# Instead of supervised learning regression where theh value of the utility function is determined,
# we estimate the utility for an investment/betting fraction based on the frequency of different outcomes.
# Instead of preparing training data beforehand, the stochastic outcomes are generated in each epoch
# As inputs we have arrays of probabilities and outcomes, as well as the investment fractions
# For outputs, we have the average utility estimated



import torch
import torch.nn as nn
import numpy as np
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

def build_nn_stoch_train_set(num_tr=100, start_pt=0.1, batch_size=1):
    x_train = np.linspace(start=start_pt, stop=10.0, num=num_tr, endpoint=True).reshape(num_tr, 1)
    y_train = get_stochastic_y(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32)

def get_stochastic_y(x_train, prob_arr, outcome_arr):
    y_train = np.zeros(len(x_train))

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

    model = build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[10, 10])

    # print(model)