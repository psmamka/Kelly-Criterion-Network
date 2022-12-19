# simple experiments with stochastic training data

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=[20, 20]):
    layer_list = []
    layer_sizes = zip([num_inputs] + list(hid_size), list(hid_size) + [num_outputs])

    for (num_in, num_out) in layer_sizes:
        layer_list.append(nn.Linear(num_in, num_out))
        layer_list.append(nn.ReLU())
    layer_list.pop()    # get rid of the last relu

    model = nn.Sequential(*layer_list)
    return model

def build_nn_determ_set(ac_range=[0, 1.0], num_ac=100):
    ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True).reshape(num_ac, 1)
    x_valid, y_valid = ac_arr, ac_arr
    return torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)

def build_nn_training_set(ac_range=[0, 1.0], num_ac=100):
    ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True).reshape(num_ac, 1)
    x_train = np.vstack((ac_arr, ac_arr))
    y_train = np.vstack((0 * ac_arr, 2 * ac_arr))
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

def get_training_dataloader(x_train, y_train, batch_size=20):
    train_dataset = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dl

def train_nn(model, x_valid, y_valid, train_dl, num_tr=100, batch_size=5, lr=0.002, 
            num_epochs=100, epoch_prog=100, train_loss_hist=None, valid_loss_hist=None):

    nn.init.xavier_uniform_(model[0].weight)

    if train_loss_hist is None or valid_loss_hist is None:
        train_loss_hist, valid_loss_hist = np.zeros(num_epochs), np.zeros(num_epochs)

    loss_fn = nn.MSELoss()  # L1Loss/MSELoss/SmoothL1Loss for probabilistic training

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adam | SGD

    for epc in range(num_epochs):
        for x_b, y_b in train_dl:   # bathces
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

def plot_results(train_loss_hist, valid_loss_hist, model, lr=1E-4, num_epochs=100, stoch_mode=True):

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, train_loss_hist)
    plt.plot(np.arange(num_epochs) + 1, valid_loss_hist)
    plt.title(f"NN Performance\nTraining Mode: {'Stochastic' if stoch_mode else 'Deterministic'}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])

    x_test, y_test = build_nn_determ_set(ac_range=[0, 1.0], num_ac=100)
    with torch.no_grad():
        y_pred = model(x_test)[:, 0]

    plt.subplot(1, 2, 2)
    plt.plot(x_test.detach().numpy()[:], y_pred.detach().numpy()[:])
    plt.plot(x_test.detach().numpy()[:], y_test.detach().numpy()[:])
    plt.title(f"Model vs Theory\nLearning Rate: {lr}")
    plt.xlabel("Input X to Model")
    plt.legend(["Model", "Theory"])

    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    stoch_mode = True  # True: Probabilistic Training | False: Deterministic Training
    if stoch_mode:
        hid_size=[15, 15]
        num_epochs = 500
        epoch_prog = 50
        num_tr=200
        batch_size = 5
        lr = 1E-4   # MSE
    else:
        hid_size=[10, 10]
        num_epochs = 100
        epoch_prog = 10
        num_tr=100
        batch_size = 2
        # lr = 0.020   # MSE
        lr = 1E-3  # L1

    train_loss_hist, valid_loss_hist = np.zeros(num_epochs), np.zeros(num_epochs)
    
    model = build_multi_hidden_nn(num_inputs=1, num_outputs=1, hid_size=hid_size)

    x_valid, y_valid = build_nn_determ_set(ac_range=[0, 1.0], num_ac=67)

    if stoch_mode:
        x_train, y_train = build_nn_training_set(ac_range=[0, 1.0], num_ac=100)
    else:
        x_train, y_train = build_nn_determ_set(ac_range=[0, 1.0], num_ac=100)
    # print(f"x_valid: {x_valid[-10:]}\n", f"y_valid: {y_valid[-10:]}\n", f"x_train: {x_train[-10:]}\n", f"y_train:{y_train[-10:]}\n")

    train_dl = get_training_dataloader(x_train, y_train, batch_size=batch_size)
    # print(f"train_dl: {train_dl} \n dataset: {train_dl.dataset}\n")

    train_nn(model, x_valid, y_valid, train_dl, num_tr=num_tr, batch_size=batch_size, lr=lr, num_epochs=num_epochs, 
    epoch_prog=epoch_prog, train_loss_hist=train_loss_hist, valid_loss_hist=valid_loss_hist)
    
    plot_results(train_loss_hist, valid_loss_hist, model, lr=lr, num_epochs=num_epochs, stoch_mode=stoch_mode)
