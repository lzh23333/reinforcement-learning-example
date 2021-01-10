#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   controller.py
@Time    :   2021/01/09 11:53:51
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   control agent and env and others.
'''
import numpy as np
import random
from tqdm import tqdm
from cat_mouse import CatAgent, BoardEnv
from utils import Label, index2pos, pos2index


class Controller(object):

    def __init__(self,
                 board_size,
                 init_state,
                 block_num=2,
                 blocks=None,
                 board=None,
                 eps=0.2,
                 mouse_move="stay"
        ):
        """
        Args:
            board_size (tuple (int, int)).
            init_state (tuple (int, int)): cat, mouse position index.
            block_num (int).
            blocks (list [(x, y)]): block position.
            mouse_move (str): mouse move pattern, ["random", "stay"].
        """
        # init board and blocks
        if board is None:
            board = Label.empty.value * np.ones(board_size)
        total = board_size[0] * board_size[1]
        self.board = board
        if blocks is None:
            blocks = random.sample(range(total), block_num + 2)
            blocks = list(filter(lambda x: x not in init_state, blocks))
            blocks = [index2pos(x, board.shape) for x in blocks[:block_num]]
        for x in blocks:
            self.board[x] = Label.block.value
        # init state    
        self.init_state = init_state
        # init Q table
        self.Q = np.random.randn(total, total, 4) * 1e-3
        # init Q[final_state] = 0
        self.Q[range(total), range(total)] = 0
        block_index = [pos2index(x, board.shape) for x in blocks]
        self.Q[block_index, :] = 0

        self.eps = eps
        self.mouse_move = mouse_move   


    def q_learning(self, lr=0.01, eta=0.7, max_iter=500):
        """perform q learning on agent.

        Returns:
            rewards (list [float]): each epsiode reward.
        """
        rewards = []
        for e in range(max_iter):
            # init cat and env agent
            cat = CatAgent(self.Q, self.init_state, self.eps)
            env = BoardEnv(self.init_state, self.board, self.mouse_move)
            reward = 0
            loop = 0
            # loop
            while not env.is_terminate():
                s0 = env.state
                action = cat.eps_greedy_action()
                s, r = env.recv(action)
                cat.recv(s)
                reward += r
                cat.Q[s0][action] = (1 - lr) * cat.Q[s0][action] +\
                                    lr * (r + eta * cat.Q[s][cat.action()])
                loop += 1
            # update
            self.Q = cat.Q
            rewards.append(reward)
            print(f"EPOCH: {e}, LOOP: {loop}, REWARD: {reward}")
    
        return rewards
    
    def epsiode(self):
        """perform search progress.

        Returns:
            state_history (list [(x0, y0), (x1, y1)]).
        """
        cat = CatAgent(self.Q, self.init_state, self.eps)
        env = BoardEnv(self.init_state, self.board, self.mouse_move)
        state_history = [env.state]
        while not env.is_terminate():
            s, _ = env.recv(cat.action())
            cat.recv(s)
            state_history.append(s)
            print(s)
        return state_history