# The simplest case of a neural network used to estimate the utility of a given betting/investment policy
# Deterministic Utility Function
# input: investment/betting fraction
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
                nn.ReLU(),
                # nn.PReLU(num_parameters=1),
                nn.Linear(hid_size, num_outputs),
                # nn.PReLU(num_parameters=1)
            )

    return model

def build_double_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[10, 10]):

    model = nn.Sequential(
                nn.Linear(num_inputs, hid_size[0]),
                nn.ReLU(),
                # nn.PReLU(num_parameters=1),
                nn.Linear(hid_size[0], hid_size[1]),
                nn.ReLU(),
                # nn.PReLU(num_parameters=1),
                nn.Linear(hid_size[1], num_outputs),
            )

    return model

def build_nn_dataset(num_tr=100, start_pt=0.1, batch_size=1):
    y_reg = -100 # regularization for very small numbers
    x_train = np.linspace(start=start_pt, stop=10.0, num=num_tr, endpoint=True).reshape(num_tr, 1)
    y_train = torch.tensor(get_log_util(x_train, x_reg=1E-100), dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    
    num_val = 32
    x_valid = np.linspace(start=start_pt, stop=10.0, num=num_val, endpoint=True).reshape(num_val, 1)
    y_valid = torch.tensor(get_log_util(x_valid, x_reg=1E-100), dtype=torch.float32)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    # print(x_valid.shape)

    train_dataset = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dl, x_valid, y_valid

def get_log_util(x, x_reg=1E-10):
    y = np.log(np.maximum(x, x_reg))
    return y

def train_nn_model_supervised(model, train_dl, x_valid, y_valid, num_tr=100, batch_size=5, lr=0.002, num_epochs=100, epoch_prog=100):

    nn.init.xavier_uniform_(model[0].weight)

    train_loss_hist = np.zeros(num_epochs)
    valid_loss_hist = np.zeros(num_epochs)

    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epc in range(num_epochs):
        for x_b, y_b in train_dl:   # batches
            pred = model(x_b)[:, 0]
            loss_tr = loss_fn(pred, y_b.squeeze())
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist[epc] += loss_tr.item()
        
        train_loss_hist[epc] /= num_tr / batch_size
        with torch.no_grad():
            pred = model(x_valid)[:, 0]
        loss_v = loss_fn(pred, y_valid.squeeze())
        valid_loss_hist[epc] += loss_v.item()

        if epc % epoch_prog == 0:
            print(f"Epoch: {epc} | Training Loss: {train_loss_hist[epc]} | Validation Loss: {valid_loss_hist[epc]}")
    
    return train_loss_hist, valid_loss_hist

def plot_results(train_loss_hist, valid_loss_hist, num_epochs=100, lr=1E-2):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, train_loss_hist)
    plt.plot(np.arange(num_epochs) + 1, valid_loss_hist)
    plt.title("Deterministi NN Training\nLogarithmic Utility")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])

    x_test = np.sort(np.random.uniform(low=0.01, high=10, size=100))
    y_test = get_log_util(x_test, x_reg=1E-10)
    with torch.no_grad():
        y_pred = model(torch.tensor(x_test.reshape(100, 1), dtype=torch.float32))[:, 0]

    plt.subplot(1, 2, 2)
    plt.plot(x_test[:], y_pred.detach().numpy()[:])
    plt.plot(x_test[:], y_test[:])
    plt.title(f"Model vs Theory\nLearning Rate: {lr}")
    plt.xlabel("Input X to Model")
    plt.ylabel("Logarithmic Utility Function")
    plt.legend(["Model", "Theory"])

    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    num_tr = 100
    start_pt = 0.01
    num_epochs = 2000
    epoch_prog = 100
    lr = 0.002
    batch_size = 2

    # model = build_single_hidden_nn(num_inputs=1, num_outputs=1, hid_size=30)

    model = build_double_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[15, 15])

    # print(model)

    train_dl, x_valid, y_valid = build_nn_dataset(num_tr=num_tr, start_pt=start_pt, batch_size=batch_size)

    train_loss_hist, valid_loss_hist = train_nn_model_supervised(model, train_dl, x_valid, y_valid, \
                                                                num_tr=num_tr, batch_size=batch_size, lr=lr, \
                                                                num_epochs=num_epochs, epoch_prog=epoch_prog)

    print(f"train loss: {train_loss_hist} \n validation loss: {valid_loss_hist}")
    
    plot_results(train_loss_hist, valid_loss_hist, num_epochs=num_epochs, lr=lr)


