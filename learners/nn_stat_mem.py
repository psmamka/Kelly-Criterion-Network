# Implementation of the Statistical Memory technique
# Continuation of, and parallel to the ε-focus technique
# 
# Statistical Memory Rationale: 
#  * Even when employing ε-focus, the nework has difficulty converging 
#       to the optmial solution due to large variance in outcomes
#  * Instead of recording individual transitions in the memory, we 
#       divide the 2-dimensional state-action (phase) space to tiles
#       (or cells)
#  * With accumulation of experiences, for each cell we keep track of the 
#       average reward, as well as the  number of samples (occurances) 
#       within each cell. 
#  * Here we use non-overlapping cells; for a general discussion of tiling
#       methods consult Sutton & Barto Reinforcement Learning (2018) Ch.9 
#  * In dynamic environments, we can shift the averaging to favor more 
#       recent samples
# 
# 
# Statistical Memory Implementation:
#   * First we initialize the 2D statistical memory grid
#       - A 2D array (matrix) of tuples with (r, n)
#       - r the average reward (utility), n the number of samples for the cell
#       - init values are (0, 0)
#   * With each additional experience/transition:
#       - first calculate the reward (diff of state/capital utilities)
#       - next calculate the memory cell coordinates
#           -- some kind of rounding of (state * N) and (action * M) numbers
#           -- N and M number of rows and columns in the mem matrix erspectively
#           -- Use the projectogram repo indexing code if helpful
#       - then update the cell n and r values
#           -- r → (n * r + r_new) / (n + 1)
#           -- n → n + 1
#       - In each training epoch, build tensor datasets and dataloaders to train
#           the nn
#           -- optionally, use n values as data weights
#   * For q-learning is is not just the current reward that we are interested,
#       - the expected reward in the next step is taken into account
#       - here we detemine the index of the memory row for the next_state
#       - within the row, we can find the highest next reward for all given actions
#       - use the highest next_reward to train the q-nn
#           -- computationally, it would be benefitial to have a column array keeping
#               track of the maximum average reward in each row
#           -- computationally, it might also help to have to matrices for 
#               r (float) and n (int)


#  Statistical Memory 2D Array schema:
#         _ _ _ _ _
#  S     |_|_|_|_|_|    
#  T     |_|_|_|_|_|
#  A     |_|_|_|_|_|    → each cell has (r_ij, n_ij)
#  T     |_|_|_|_|_|    → alternatively, separate matrices for r (float) and n (integer)
#  E     |_|_|_|_|_|
#  S     
# (i)     ACTIONS (j)
# 
# 