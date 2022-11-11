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


if __name__ == "__main__":
    plot_lin_vs_log_util()
    plot_sqrt_util()
