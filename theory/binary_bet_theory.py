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
# E(s | n-step) / s = Σ n!/k!/(n-k)! * win_pr^k * loss_pr^(n-k) * (1 + bet_fr * win_fr)^k * (1 - bet_fr * loss_fr)^(n-k)          
# with summation over k = 0 to n, each k corresponding to k wins and n-k losses, and
# n!/k!/(n-k)! corresponding to number of possibilities of k wins in a sequence of n trades.


import numpy as np
from scipy.special import comb

def bet_n_times_outcome(n, bet_fr, win_pr, win_fr=1.0, loss_fr=1.0):
    if type(bet_fr) is np.ndarray:
        outcomes = np.zeros(bet_fr.shape)
    else:
        outcomes = 0
    
    for k in range(n + 1):
        outcomes += comb(n, k, exact=True) * win_pr**k * (1 - win_pr)**(n-k) * \
                        (1 + bet_fr * win_fr)**k * (1 - bet_fr * loss_fr)**(n-k)
    
    return outcomes


if __name__ == "__main__":
    bet_fr = np.linspace(start=0.0, stop=1.0, num=11, endpoint=True)
    steps = np.arange(start=1, stop=11, step=1)
    results = np.zeros((steps.size, bet_fr.size))

    win_pr = 0.6
    win_fr = 1.0
    loss_fr = 1.0

    for idx, n in enumerate(steps):
        results[idx, :] = bet_n_times_outcome(n, bet_fr, win_pr, win_fr, loss_fr)
    
    import matplotlib.pyplot as plt

    plt.figure()
    for idx, _ in enumerate(steps):
        plt.plot(bet_fr, results[idx, :])    
    plt.show()

