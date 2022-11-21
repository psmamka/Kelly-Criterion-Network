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
    layer_list.pop()    # get rid of the last relu

    model = nn.Sequential(*layer_list)
    return model

def build_nn_stoch_train_set(prob_arr, outcome_arr, start_pt=0.0, stop_pt=0.99, num_tr=100, batch_size=1):
    # Here x_train are the investment fractions
    x_train = np.linspace(start=start_pt, stop=stop_pt, num=num_tr, endpoint=True).reshape(num_tr, 1)
    y_train = get_stoch_outcome_y(x_train, prob_arr, outcome_arr)

    # print(x_train, y_train)

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
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
    # print(rand_arr, rand_arr.sum())
    # cumulative probabilities
    p_cum = np.cumsum(p_arr)
    # print(p_cum)
    # outcome indices:
    for idx, x in enumerate(x_in):
        p_idx = np.nonzero(p_cum > rand_arr[idx])[0][0]
        # investment outcome based on investment fraction and the stochastic probabilities
        y_train[idx] = (1 - x) + x * o_arr[p_idx]
    # print(y_train)
    # convert to log-utility
    y_train = get_log_util(y_train, x_reg=1E-20)
    return y_train

def get_determ_outcome_y(x_in, prob_arr, outcome_arr):
    y = np.zeros(len(x_in))
    p_arr, o_arr = np.array(prob_arr), np.array(outcome_arr)
    for idx, x in enumerate(x_in):
        # print((1-x) * np.ones(len(o_arr)) + x * o_arr)
        y[idx] = np.sum(p_arr * get_log_util((1-x) * np.ones(len(o_arr)) + x * o_arr, x_reg = 1E-20))
    return y

def build_nn_valid_set(prob_arr, outcome_arr, start_pt=0.0, stop_pt=0.99, num_val=35):
    x_valid = np.linspace(start=start_pt, stop=stop_pt, num=num_val, endpoint=True).reshape(num_val, 1)
    y_valid = get_determ_outcome_y(x_valid, prob_arr, outcome_arr)

    print(x_valid, y_valid)
    return torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)

def get_log_util(x, x_reg=None, y_reg=None):
    if x_reg is not None:
        y = np.log(np.maximum(x, x_reg))
    elif y_reg is not None:
        y = np.maximum(np.log(x), y_reg)
    else:
        y = np.log(x)
    return y

def train_nn_probabilistic(model, x_valid, y_valid, prob_arr, outcome_arr, num_tr=100, batch_size=5, lr=0.002, num_epochs=100):

    nn.init.xavier_uniform_(model[0].weight)

    train_loss_hist = np.zeros(num_epochs)
    valid_loss_hist = np.zeros(num_epochs)

    loss_fn = nn.SmoothL1Loss()  # L1/MSE/BCE/SmoothL1 loss for probabilistic training
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # train_dl = build_nn_stoch_train_set(prob_arr, outcome_arr,start_pt=0.0, stop_pt=0.99, num_tr=num_tr, batch_size=batch_size)
    # x_train, y_train = build_nn_valid_set(prob_arr, outcome_arr, start_pt=0.0, stop_pt=0.99, num_val=100)
    # train_dataset = TensorDataset(x_train, y_train)
    # train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epc in range(num_epochs):
        # in each epoch new sampling of the training dataset is created
        train_dl = build_nn_stoch_train_set(prob_arr, outcome_arr,start_pt=0.0, stop_pt=0.99, num_tr=num_tr, batch_size=batch_size)
        for x_b, y_b in train_dl:   # bathces
            pred = model(x_b)[:, 0]
            loss_tr = loss_fn(pred, y_b.squeeze())
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist[epc] += loss_tr.item()

        train_loss_hist[epc] /= num_tr / batch_size
        pred = model(x_valid)[:, 0]
        loss_v = loss_fn(pred, y_valid.squeeze())
        valid_loss_hist[epc] += loss_v.item()

    return train_loss_hist, valid_loss_hist

def plot_results(train_loss_hist, valid_loss_hist, model, prob_arr, outcome_arr, num_epochs=100):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, train_loss_hist)
    plt.plot(np.arange(num_epochs) + 1, valid_loss_hist)
    plt.title("NN Performance\nProbabilistic Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])

    x_test = np.sort(np.random.uniform(low=0.01, high=0.9, size=100))
    y_test = get_determ_outcome_y(x_test, prob_arr, outcome_arr)
    y_pred = model(torch.tensor(x_test.reshape(100, 1), dtype=torch.float32))[:, 0]

    plt.subplot(1, 2, 2)
    plt.plot(x_test[:], y_pred.detach().numpy()[:])
    plt.plot(x_test[:], y_test[:])
    plt.title("Model vs Theory")
    plt.xlabel("Input X to Model")
    plt.ylabel("Logarithmic Utility Function")
    plt.legend(["Model", "Theory"])

    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    prob_arr = [0.4, 0.6]
    outcome_arr = [0.0, 2.0]
    num_epochs = 100
    num_tr=500
    batch_size = 10
    lr = 0.0002

    model = build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[20, 20])

    print(model)

    x_valid, y_valid = build_nn_valid_set(prob_arr, outcome_arr, start_pt=0.0, stop_pt=0.99, num_val=35)

    train_loss_hist, valid_loss_hist = train_nn_probabilistic(model, x_valid, y_valid, prob_arr, outcome_arr, \
                                                            num_tr=num_tr, batch_size=batch_size, lr=lr, num_epochs=num_epochs)

    print("Probabilistic Neural Network Training.")
    print(f"train loss: {train_loss_hist} \n validation loss: {valid_loss_hist}")

    plot_results(train_loss_hist, valid_loss_hist, model, prob_arr, outcome_arr, num_epochs=num_epochs)

    # # testing stochastic outcome: 50-50 Double or Nothing log utility, 10 times.
    # y = get_stoch_outcome_y(np.ones(10), [0.5, 0.5], [2.0, 0.0])
    # print(y)

    # print(model)