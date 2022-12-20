# 2D version of the stoch_nn_learn

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def build_multi_hidden_nn(num_inputs=2, num_outputs=1, hid_size=[20, 20]):
    layer_list = []
    layer_sizes = zip([num_inputs] + list(hid_size), list(hid_size) + [num_outputs])

    for (num_in, num_out) in layer_sizes:
        layer_list.append(nn.Linear(num_in, num_out))
        layer_list.append(nn.ReLU())
    layer_list.pop()    # get rid of the last relu

    model = nn.Sequential(*layer_list)
    return model

def build_nn_determ_set(st_range=[0.0, 1.0], ac_range=[0, 1.0], num_st=10, num_ac=10, rr=1.0):
    # rr: reward ratio, i.e. expected reward per amount risked: reward = rr * st * ac
    st_arr = np.linspace(start=st_range[0], stop=st_range[1], num=num_st, endpoint=True)
    ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True) # .reshape(num_ac, 1)
    ss, aa = np.meshgrid(st_arr, ac_arr, indexing='ij')
    # x_valid, y_valid = ac_arr, ac_arr
    x_valid = np.array(list(zip(ss.reshape(num_st * num_ac,), aa.reshape(num_st * num_ac,))))
    y_valid = (aa * ss).reshape((num_st * num_ac, 1)) * rr   # y = st * ac
    print(f"x_valid.shape: {x_valid.shape} | y_valid.shape: {y_valid.shape}" )
    return torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)

def build_nn_training_set_sym(st_range=[0.0, 1.0], ac_range=[0, 1.0], num_st=10, num_ac=10):
    # symmetric division of points between 0 and twice expected reward 
    st_arr = np.linspace(start=st_range[0], stop=st_range[1], num=num_st, endpoint=True)
    ac_arr = np.linspace(start=ac_range[0], stop=ac_range[1], num=num_ac, endpoint=True) # .reshape(num_ac, 1)
    ss, aa = np.meshgrid(st_arr, ac_arr, indexing='ij')

    x_train = np.array(list(zip(ss.reshape(num_st * num_ac,), aa.reshape(num_st * num_ac,))))
    y_train = (aa * ss).reshape((num_st * num_ac, 1))   # y = st * ac
    x_train = np.vstack((x_train, x_train))
    y_train = np.vstack((0 * y_train, 2 * y_train))
    print(f"x_train.shape: {x_train.shape} | y_train.shape: {y_train.shape}" )
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

def build_nn_training_set_stoch(st_range=[0.0, 1.0], ac_range=[0, 1.0], num_samp=200, prob_win=0.5):
    # fully stochastic selection of state, action and reward (between 0 and 2 * expeted reward)
    rng = np.random.default_rng(1)
    st_arr = rng.uniform(low=st_range[0], high=st_range[1], size=num_samp)
    ac_arr = rng.uniform(low=ac_range[0], high=ac_range[1], size=num_samp)
    x_train = np.array(list(zip(st_arr, ac_arr)))   # print(x_train[-20:])
    y_train = np.array(list(map(
                                lambda x: x[0] * x[1] * 2 if rng.uniform() < prob_win else 0, x_train
                                ))).reshape((num_samp, 1))  # print(y_train[-20:])
    print(f"x_train.shape: {x_train.shape} | y_train.shape: {y_train.shape}" )
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

def get_training_dataloader(x_train, y_train, batch_size=100):
    train_dataset = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dl

def train_nn(model, x_valid, y_valid, train_dl, num_tr=200, batch_size=50, lr=0.001, 
            num_epochs=1000, epoch_prog=100, train_loss_hist=None, valid_loss_hist=None):

    nn.init.xavier_uniform_(model[0].weight)
    # for layer in model:    # initialize linear layers
    #     if type(layer) == nn.Linear: nn.init.xavier_uniform_(layer.weight)

    if train_loss_hist is None or valid_loss_hist is None:
        train_loss_hist, valid_loss_hist = np.zeros(num_epochs), np.zeros(num_epochs)

    loss_fn = nn.MSELoss()  # L1Loss/MSELoss/SmoothL1Loss for probabilistic training

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # SGD | Adam

    for epc in range(num_epochs):
        for x_b, y_b in train_dl:   # bathces
            pred = model(x_b)[:, 0]
            loss_tr = loss_fn(pred, y_b.squeeze())
            loss_tr.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_hist[epc] += loss_tr.item()

        train_loss_hist[epc] /= len(train_dl.dataset) / batch_size
        with torch.no_grad():
            pred = model(x_valid)[:, 0]
        loss_v = loss_fn(pred, y_valid.squeeze())
        valid_loss_hist[epc] += loss_v.item()

        if epc % epoch_prog == 0:
            print(f"Epoch: {epc} | Training Loss: {train_loss_hist[epc]} | Validation Loss: {valid_loss_hist[epc]}")

    return train_loss_hist, valid_loss_hist

def plot_results(train_loss_hist, valid_loss_hist, model, lr=1E-4, num_epochs=100, stoch_mode=True):

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(num_epochs) + 1, train_loss_hist)
    plt.plot(np.arange(num_epochs) + 1, valid_loss_hist)
    plt.title(f"NN Performance\nTraining Mode: {'Stochastic' if stoch_mode else 'Deterministic'}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training", "Validation"])
    plt.show()

    num_st = 10
    num_ac = 10
    x_test, y_test = build_nn_determ_set(st_range=[0.0, 1.0], ac_range=[0, 1.0], num_st=num_st, num_ac=num_ac)
    with torch.no_grad():
        y_pred = model(x_test)[:, 0]
    
    y_2d_test =  y_test.detach().numpy().reshape((num_st, num_ac))
    y_2d_pred =  y_pred.detach().numpy().reshape((num_st, num_ac))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(y_2d_test)
    plt.colorbar()
    plt.title(f"Target Contour")
    # plt.xlabel("Input X to Model")
    # plt.legend(["Model", "Theory"])
    plt.subplot(1, 2, 2)
    plt.contourf(y_2d_pred)
    plt.colorbar()
    plt.title(f"Predicted Contour")
    plt.show()

    # diagonal (a = s) comparison plot
    y_diag1d_test = list(map(lambda i: y_2d_test[i, i], np.arange(num_st)))
    y_diag1d_pred = list(map(lambda i: y_2d_pred[i, i], np.arange(num_st)))
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(num_st), y_diag1d_test, np.arange(num_st), y_diag1d_pred)
    plt.legend(['Target', 'Model'])
    plt.title(f"Diagonal Elements: Model vs Target")
    plt.show()


if __name__ == "__main__":
    
    np.random.seed(1)
    torch.manual_seed(1)

    stoch_mode = True   # True: Probabilistic Training | False: Deterministic Training
    sym_mode = False     # True: symmetric division of points between 0 and 2 * expected reward | False: full stochastic
    if stoch_mode:
        hid_size=[10, 10]
        num_epochs = 10_000 if sym_mode else 20_000
        epoch_prog = 500
        num_tr=200
        batch_size = 200 if sym_mode else 200
        lr = 3E-4 if sym_mode else 2E-5
    else:
        hid_size=[10, 10]
        num_epochs = 500
        epoch_prog = 50
        num_tr=100
        batch_size = 10
        # lr = 0.020   # MSE
        lr = 3E-4  # L1

    train_loss_hist, valid_loss_hist = np.zeros(num_epochs), np.zeros(num_epochs)
    
    model = build_multi_hidden_nn(num_inputs=2, num_outputs=1, hid_size=hid_size)

    x_valid, y_valid = build_nn_determ_set(st_range=[0, 1.0], ac_range=[0, 1.0], num_st=9, num_ac=9, rr=1.0)

    if stoch_mode:
        if sym_mode:
            x_train, y_train = build_nn_training_set_sym(st_range=[0, 1.0], ac_range=[0, 1.0], num_st=10, num_ac=10, sym_mode=sym_mode)
        else:
            x_train, y_train = build_nn_training_set_stoch(st_range=[0.0, 1.0], ac_range=[0, 1.0], num_samp=200, prob_win=0.5)
    else:
        x_train, y_train = build_nn_determ_set(st_range=[0, 1.0], ac_range=[0, 1.0], num_st=10, num_ac=10)
    # print(f"x_valid: {x_valid[-10:]}\n", f"y_valid: {y_valid[-10:]}\n", f"x_train: {x_train[-10:]}\n", f"y_train:{y_train[-10:]}\n")

    train_dl = get_training_dataloader(x_train, y_train, batch_size=batch_size)
    # print(f"train_dl: {train_dl} \n dataset: {train_dl.dataset}\n len: {len(train_dl.dataset)}")

    train_nn(model, x_valid, y_valid, train_dl, num_tr=num_tr, batch_size=batch_size, lr=lr, num_epochs=num_epochs, 
    epoch_prog=epoch_prog, train_loss_hist=train_loss_hist, valid_loss_hist=valid_loss_hist)
    
    plot_results(train_loss_hist, valid_loss_hist, model, lr=lr, num_epochs=num_epochs, stoch_mode=stoch_mode)
