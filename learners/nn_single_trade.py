# The simplest case of a neural network used to estimate the utility of a given betting/investment policy
# 
# input: investment fraction
# output: estimated utility for investment outcome

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


def build_single_hidden_nn(num_inputs=1, num_outputs=1, hid_size=20):
    model = nn.Sequential(
                nn.Linear(num_inputs, hid_size),
                nn.PReLU(num_parameters=1),
                nn.Linear(hid_size, num_outputs)
            )

    nn.init.xavier_uniform_(model[0].weight)

    return model

def build_nn_dataset():
    y_reg = -100 # regularization for very small numbers
    x_train = torch.tensor(np.linspace(0, 10.0, 101, endpoint=True), dtype=torch.float32)
    y_train = torch.tensor(get_log_util(x_train, x_reg=1E-100), dtype=torch.float32)
    
    x_valid = torch.tensor(np.linspace(0, 1.0, 33, endpoint=True), dtype=torch.float32)
    y_valid = torch.tensor(get_log_util(x_valid, x_reg=1E-100), dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    return train_dl, x_valid, y_valid

def get_log_util(x, x_reg=1E-10):
    y = np.log(np.maximum(x, x_reg))
    return y

def train_nn_model_supervised(model, train_dl, x_valid, y_valid, lr=0.001, num_epochs=100):

    train_loss_hist = np.zeros(num_epochs)
    valid_loss_hist = np.zeros(num_epochs)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epc in range(num_epochs):
        for x_b, y_b in train_dl:   # batches
            pred = model(x_b)
            loss_tr = loss_fn(pred, y_b)
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist[epc] += loss_tr
        
        pred = model(x_valid)
        loss_v = loss_fn(pred, y_valid)
        valid_loss_hist[epc] += loss_v
    
    return train_loss_hist, valid_loss_hist

def plot_results(train_loss_hist, valid_loss_hist, num_epochs=100):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, train_loss_hist, np.arange(num_epochs) + 1, valid_loss_hist)
    plt.title("Single Hidden Layer NN\nLogarithmic Util Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])

    x_test = np.random.uniform(low=0, high=10, size=100)
    y_test = get_log_util(x_test, x_reg=1E-10)
    y_pred = model(x_test)

    plt.subplot(1, 2, 2)
    plt.plot(x_test, y_pred, x_test, y_test)
    plt.title("Model vs Theory")
    plt.xlabel("Input X to Model")
    plt.ylabel("Logarithmic Utility Function")
    plt.legend(["Model", "Theory"])

    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    model = build_single_hidden_nn(num_inputs=1, num_outputs=1, hid_size=20)

    train_dl, x_valid, y_valid = build_nn_dataset()

    train_loss_hist, valid_loss_hist = train_nn_model_supervised(model, train_dl, x_valid, y_valid, lr=0.001, num_epochs=100)

    plot_results(train_loss_hist, valid_loss_hist, num_epochs=100)


