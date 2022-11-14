# In each step, we have two outcome:
# (1) win: with probability win_pr, state (capital) is increased by:
#  s ↦ s + (bet_fr * s) * win_fr
# 
# (2) loss: with probability loss_pr (= 1 - win_pr), the state idecreased by:
#  s ↦ s - (bet_fr * s) * loss_fr
# 
#  After a single step, the expected state can be written as:
# 
#  E(s | 1-step) = win_pr *  (s + (bet_fr * s) * win_fr) + loss_pr * (s - (bet_fr * s) * loss_fr)
#                = s + win_pr * (bet_fr * s) * win_fr - loss_pr * (bet_fr * s) * loss_fr
# 
# Normalizing the expectation value by the initial capital s, re-arranging terms:
# 
# E(s | 1-step) / s = win_pr * (1 + bet_fr * win_fr) + loss_pr * (1 - bet_fr * loss_fr)
# 
# For a n-step trial, with compounding of returns, we sum over all possible combinations of gain/loss sequences:
# 
# E(s | n-step) / s = Σ n!/k!/(n-k)!  *  win_pr^k * loss_pr^(n-k)  *  (1 + bet_fr * win_fr)^k * (1 - bet_fr * loss_fr)^(n-k)          
#                       ------------    --------------------------    ------------------------------------------------------
#                       ↑ num combs ↑   ↑ prob. of a combination ↑       ↑  return of the k-win, (n-k)-loss combination  ↑
# 
# with summation over k = 0 to n, each k corresponding to k wins and n-k losses, and
# n!/k!/(n-k)! corresponding to number of possibilities of k wins in a sequence of n trades.
# 
# When optimizing the logarithm of returns, the fomula is updated to:
# 
# E(log(s) | n-step) - log(s) = Σ n!/k!/(n-k)! * win_pr^k * loss_pr^(n-k) * log[ (1 + bet_fr * win_fr)^k * (1 - bet_fr * loss_fr)^(n-k) ]
# 
# Next we plot expected returns for another utility function, namely the square root of capital. In this case, the optimal
# betting fraction ends up being somewhere in between the kelly criterion (for log expectation) and the full betting fraction
# for linear expectation.
# 
# Another spacial case of interest is the utility function for the expected profits for a fund manager, which we refer to
# as HFM(x). It is customary for private funds to charge a maintanence fee as a fixed ratio of the capital managed, as well
# as a larger percentage of the annual profits/performance, combined with certain "high-water mark" condition. An interesting 
# case would be when HFM results in trades despite negative expectancy. 

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def bet_n_times_outcome(n, bet_fr, win_pr, win_fr=1.0, loss_fr=1.0, log_return=False):
    if type(bet_fr) is np.ndarray:
        outcomes = np.zeros(bet_fr.shape)
    else:
        outcomes = 0
    
    if log_return:
        for k in range(n + 1):
            outcomes += comb(n, k, exact=True) * win_pr**k * (1 - win_pr)**(n-k) * \
                            np.log((1 + bet_fr * win_fr)**k * (1 - bet_fr * loss_fr)**(n-k))
    else:
        for k in range(n + 1):
            outcomes += comb(n, k, exact=True) * win_pr**k * (1 - win_pr)**(n-k) * \
                            (1 + bet_fr * win_fr)**k * (1 - bet_fr * loss_fr)**(n-k)
    
    return outcomes

def bet_n_times_outcome_ufunc(n, bet_fr, win_pr, win_fr=1.0, loss_fr=1.0, util_func=lambda x: x):
    '''generalized form of bet_n_times_outcome with a utility function passed in as an input'''
    if type(bet_fr) is np.ndarray:
        outcomes = np.zeros(bet_fr.shape)
    else:
        outcomes = 0

    for k in range(n + 1):
        outcomes += comb(n, k, exact=True) * win_pr**k * (1 - win_pr)**(n-k) * \
                        util_func((1 + bet_fr * win_fr)**k * (1 - bet_fr * loss_fr)**(n-k))
    
    return outcomes

def plot_lin_vs_log_util():
    bet_fr = np.linspace(start=0.0, stop=1.0, num=50, endpoint=False)
    steps = np.array([1, 2, 3, 5, 10])
    results_lin = np.zeros((steps.size, bet_fr.size))
    results_log = np.zeros((steps.size, bet_fr.size))

    win_pr = 0.6
    win_fr = 1.0
    loss_fr = 1.0

    for idx, n in enumerate(steps):
        results_lin[idx, :] = bet_n_times_outcome(n, bet_fr, win_pr, win_fr, loss_fr, log_return=False)
        results_log[idx, :] = bet_n_times_outcome(n, bet_fr, win_pr, win_fr, loss_fr, log_return=True)

    plt.figure(figsize=(11, 5))
    for idx, n in enumerate(steps):
        plt.subplot(1, 2, 1)
        plt.plot(bet_fr, results_lin[idx, :])
        plt.subplot(1, 2, 2)
        plt.plot(bet_fr, results_log[idx, :], label=f"n={n}")    
        # if idx==0 or idx == len(steps) - 1:
        #     plt.legend(str(n))

    plt.subplot(1, 2, 1)
    plt.ylim((0.8, 2))
    plt.ylabel("Linear Returns Expectation")
    plt.xlabel("Betting Fractiong")
    plt.subplot(1, 2, 2)
    plt.ylim((-0.3, 0.3))
    plt.ylabel("Logarithmic Returns Expectation")
    plt.xlabel("Betting Fractiong")
    plt.legend()
    plt.show()

def plot_sqrt_util():
    bet_fr = np.linspace(start=0.0, stop=1.0, num=50, endpoint=True)
    steps = np.array([1, 2, 3, 5, 10])
    results = np.zeros((steps.size, bet_fr.size))

    win_pr = 0.6
    win_fr = 1.0
    loss_fr = 1.0

    for idx, n in enumerate(steps):
        results[idx, :] = bet_n_times_outcome_ufunc(n, bet_fr, win_pr, win_fr, loss_fr, util_func=lambda x: np.sqrt(x))
    
    plt.figure()
    for idx, n in enumerate(steps):
        plt.plot(bet_fr, results[idx, :], label=f"n={n}")

    plt.ylim((0.5, 1.4))
    plt.ylabel("Square Root Returns Expectation")
    plt.xlabel("Betting Fractiong")
    plt.legend()
    plt.show()

def hfm_util(x_new, x_old, hwm=None, fix_ratio=0.01, perf_ratio=0.10):
    '''the hedge fund manager's utility function, for increasing capital from x_old to x_new, in the presence of
    a "high-water-mark", with given fixed ratio fees and performance ratio charges applied to invested funds'''
    if hwm is None:
        pvt = x_old # pivot point
    else:
        pvt = max(hwm, x_old)
    
    total_fees = fix_ratio * x_new + perf_ratio * np.maximum(np.zeros(len(x_new)), x_new - pvt)
    return total_fees

def hfc_util(x_new, x_old, hwm=None, fix_ratio=0.01, perf_ratio=0.10):
    '''A hedge-fund client's utility function; linear returns and taking into effect the fixed and performance
    fees. Can be combined with any other utility func.'''
    if hwm is None:
        pvt = x_old # pivot point
    else:
        pvt = max(hwm, x_old)
    
    total_fees = fix_ratio * x_new + perf_ratio * np.maximum(np.zeros(len(x_new)), x_new - pvt)
    return x_new - x_old - total_fees

def plot_hf_util_single_period():
    bet_fr = np.linspace(start=0.0, stop=1.0, num=50, endpoint=True)

    win_fr = 1.0
    loss_fr = 1.0
    util_func_manager=lambda x: hfm_util(x, 1, 1, fix_ratio=0.01, perf_ratio=0.10)
    util_func_client=lambda x: hfc_util(x, 1, 1, fix_ratio=0.01, perf_ratio=0.10)

    # results for manager and client, with positive and negative "edge"
    res_manager_pos = bet_n_times_outcome_ufunc(1, bet_fr, 0.6, win_fr, loss_fr, util_func=util_func_manager)
    res_client_pos = bet_n_times_outcome_ufunc(1, bet_fr, 0.6, win_fr, loss_fr, util_func=util_func_client)
    res_manager_neg = bet_n_times_outcome_ufunc(1, bet_fr, 0.4, win_fr, loss_fr, util_func=util_func_manager)
    res_client_neg = bet_n_times_outcome_ufunc(1, bet_fr, 0.4, win_fr, loss_fr, util_func=util_func_client)

    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bet_fr, res_manager_pos, bet_fr, res_client_pos)
    plt.xlabel("Investment Fractiong")
    plt.ylabel("HF Returns Expectation")
    plt.title("HF Returns:\n60% Win Double-or-Nothing ")
    plt.legend(["Manager", "Client"])
    plt.subplot(1, 2, 2)
    plt.plot(bet_fr, res_manager_neg, bet_fr, res_client_neg)
    plt.xlabel("Investment Fractiong")
    plt.ylabel("HF Returns Expectation")
    plt.title("HF Returns:\n40% Win Double-or-Nothing ")
    plt.legend(["Manager", "Client"])
    plt.show()

def to_basis_2_arr(n, l=None):
    #  number, array length
    if l is None: l = int(np.ceil(np.log2(n)) + 1)
    arr = np.zeros(l, np.int64)
    for i in range(l):
        arr[i] = int(n % 2)
        n = int(n / 2)
    return np.flip(arr)

def plot_hf_util_multi_period(num):
    fix_ratio, perf_ratio = 0.01, 0.10
    win_fr, loss_fr = 1.0, 1.0
    win_pr = 0.6

    bet_fr = np.linspace(start=0.0, stop=1.0, num=50, endpoint=True)

    results_manager_pos = np.zeros(bet_fr.size) # positive expectancy: 60% win in double-or-nothing
    results_client_pos = np.zeros(bet_fr.size)
    results_manager_neg = np.zeros(bet_fr.size) # negative expectancy: 40% win in double-or-nothing
    results_client_neg = np.zeros(bet_fr.size)

    seq = np.zeros(num)             # a sequence of wins and losses
    for comb in range(2**num):      # all possible win-loss sequences of length num
        seq = to_basis_2_arr(comb, num)
        cap, hwm, manager_gain, client_gain = np.ones(bet_fr.size), np.ones(bet_fr.size), np.zeros(bet_fr.size), np.zeros(bet_fr.size)
        for i in range(len(seq)):
            if seq[i] == 1:  # win
                new_cap = cap * (np.ones(bet_fr.size) + win_fr * bet_fr)
                fees = fix_ratio * new_cap + perf_ratio * np.maximum(new_cap - hwm, np.zeros(bet_fr.size))
                new_cap -= fees
            else:   # loss
                new_cap = cap * (np.ones(bet_fr.size) - loss_fr * bet_fr)
                fees = fix_ratio * new_cap
                new_cap -= fees
            
            manager_gain += fees
            client_gain += (new_cap - cap)

            hwm = np.maximum(hwm, new_cap)
            cap = new_cap
        # sequence of win loss probabilities
        seq_prob_pos = win_pr**seq.sum() * (1 - win_pr)**(seq.size - seq.sum())
        seq_prob_neg = (1 - win_pr)**seq.sum() * win_pr**(seq.size - seq.sum())
        # update results, normalizing to sequence probabilities
        results_manager_pos += manager_gain * seq_prob_pos
        results_client_pos += client_gain * seq_prob_pos
        results_manager_neg += manager_gain * seq_prob_neg
        results_client_neg += client_gain * seq_prob_neg

    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bet_fr, results_manager_pos, bet_fr, results_client_pos)
    plt.xlabel("Investment Fractiong")
    plt.ylabel("HF Returns Expectation")
    plt.title(f"HF Returns for {num} Trade(s):\n60% Win Double-or-Nothing ")
    plt.legend(["Manager", "Client"])
    plt.subplot(1, 2, 2)
    plt.plot(bet_fr, results_manager_neg, bet_fr, results_client_neg)
    plt.xlabel("Investment Fractiong")
    plt.ylabel("HF Returns Expectation")
    plt.title(f"HF Returns for {num} Trade(s):\n40% Win Double-or-Nothing ")
    plt.legend(["Manager", "Client"])
    plt.show()


if __name__ == "__main__":
    plot_lin_vs_log_util()
    plot_sqrt_util()
    plot_hf_util_single_period()
    plot_hf_util_multi_period(num = 5)