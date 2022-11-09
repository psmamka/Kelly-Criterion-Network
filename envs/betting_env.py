# The betting environmnet. The API is somewhat similar to that of OpenAI geymnasium, but simpler

import numpy as np
# import random

class BettingEnvBinary():
    def __init__(self, win_pr, loss_pr, win_fr, loss_fr, 
                    start_cap=1000, max_cap=1E6, min_cap=1, max_steps=None,
                    log_returns=False, log_regul=-10):
        
        if (win_pr + loss_pr != 1.0):
            pr_sum = win_pr + loss_pr
            win_pr , loss_pr = (win_pr / pr_sum, loss_pr / pr_sum)
        
        self.win_pr = win_pr
        self.loss_pr = loss_pr
        self.win_fr = win_fr
        self.loss_fr = loss_fr
        self.start_cap = start_cap
        self.max_cap = max_cap
        self.min_cap = min_cap
        self.max_steps = max_steps

        self.log_returns = log_returns
        self.log_regul = log_regul

        self.cur_cap = start_cap
        self.cur_step = 0
        self.terminated = False
    
    def reset(self, start_cap=None):
        if start_cap is None:
            self.cur_cap = self.start_cap
        else:
            self.cur_cap = start_cap
        self.cur_step = 0
        self.terminated = False
        return self.cur_cap

    def step(self, bet_size):
        err_msg_bet_sz = f"The betting size {bet_size} should be between 0.0 and current capital {self.cur_cap}"
        assert 0 <= bet_size <= self.cur_cap, err_msg_bet_sz

        err_msg_terminated = "You are calling `step` after the episode termination."
        assert not self.terminated, err_msg_terminated

        if np.random.uniform() < self.win_pr:
            # win
            reward = bet_size * self.win_fr
        else:
            # loss
            reward = -1 * bet_size * self.loss_fr
        
        self.cur_cap += reward
        self.cur_step += 1

        self._check_termination()

        if self.log_returns:
            log_reward = self._get_log_reward(reward)
            return self.cur_cap, log_reward, self.terminated
        else:
            return self.cur_cap, reward, self.terminated

    def _check_termination(self):
        self.terminated = bool(
            (self.max_cap is not None and self.cur_cap > self.max_cap) or 
            (self.min_cap is not None and self.cur_cap < self.min_cap) or 
            (self.max_steps is not None and self.cur_step >= self.max_steps)
        )

    def _get_log_reward(self, r):
        '''calculate logarithmic reward based on linear reward and current updated capital:
        log(S_n+1) - log(S_n) = log(S_n+1 / S_n) = log((S_n + r) / S_n) = log(S_n+1 / (S_n+1 - r))'''
        
        if self.cur_cap < 1E-10: return self.log_regul

        log_rw = np.log(self.cur_cap / (self.cur_cap - r))

        if log_rw == np.nan or log_rw == -np.Inf or log_rw < self.log_regul:
            return self.log_regul
        else:
            return log_rw


        