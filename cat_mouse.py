#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cat_mouse.py
@Time    :   2021/01/07 17:12:23
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   cat mouse catch example and corresponding env and agent.
'''
import random
import numpy as np
from utils import (
    move_on_board,
    index2pos,
    pos2index,
    Label,
    manhattan
)


class CatAgent(object):
    """Cat Agent for example.

    Attributes:
        Q (np.ndarray): [n_states, n_states, 4], state is position's index.
        state (int): indicate state agent face.
    """

    def __init__(self, Q, state, eps=0.2):
        self.Q = Q
        self.state = state
        self.eps = eps

    def recv(self, state):
        self.state = state

    def action(self):
        """use Q to find best action.

        Args:
            forbid (int): if specified, [0-3], agent will not take the forbid action.
        Returns:
            action (int).
        """
        return np.argmax(self.Q[self.state])

    def eps_greedy_action(self):
        best_action = self.action()
        if random.random() > self.eps:
            return best_action
        else:
            return random.choice([x for x in range(4) if x != best_action])


class Mouse(object):
    """Mouse object, move in a MDP manner.

    Attributes:
        pos (tuple (x, y)): mouse location.
        move: move method for mouse.
    """

    def __init__(self, pos, board, ):
        self.pos = pos
        self.board = board

    def move(self, cat_pos=None, method="random"):
        """random move.

        Args:
            method (str): support "random", "stay", "away".
        """
        if method == "random":
            action = random.randint(0, 3)
            pos = move_on_board(self.pos, action, self.board.shape)
            if self.board[pos] != Label.block.value:
                self.pos = pos
        elif method == "away" and random.random() > 0.7:
            next_pos = [move_on_board(self.pos, a, self.board.shape)
                        for a in range(4)]
            next_pos = [x for x in next_pos
                        if self.board[x] != Label.block.value and x != cat_pos]
            scores = [manhattan(p, cat_pos) for p in next_pos]
            if scores != []:
                self.pos = next_pos[np.argmax(scores)]
        return self.pos


class BoardEnv(object):
    """Cat Catch Mouse Env.

    Attributes:
        state (tuple (int, int)): cat and mouse position.
        board (np.ndarray): board map, 0 means empty, 1 means block.
        recv: recieve action and return updated state and reward.
    """

    def __init__(self, state, board, mouse_pattern, reward):
        self.state = state
        self.board = board
        self.mouse = Mouse(index2pos(state[1], self.board.shape), self.board)
        self.mouse_move = mouse_pattern
        self.reward_pattern = reward
        self.count = 0

    def recv(self, action):
        """
        Args:
            action (int): [0--3].
        """
        self.count += 1
        cat_pos = index2pos(self.state[0], self.board.shape)
        cat_pos = move_on_board(cat_pos, action, self.board.shape)
        mouse_pos = self.mouse.move(
            cat_pos=cat_pos,
            method=self.mouse_move
        )
        self.state = (
            pos2index(cat_pos, self.board.shape),
            pos2index(mouse_pos, self.board.shape)
        )
        return self.state, self.reward(self.reward_pattern)

    def is_terminate(self):
        catch = self.state[0] == self.state[1]
        block = (
            self.board[index2pos(
                self.state[0], self.board.shape)] == Label.block.value
        )
        return catch or block

    def reward(self, method="basic"):
        if method == "dist":
            # reward based on distance
            cat_pos = index2pos(self.state[0], self.board.shape)
            if (self.board[index2pos(self.state[0], self.board.shape)]
                == Label.block.value):
                return -1000
            mouse_pos = index2pos(self.state[1], self.board.shape)
            dist = manhattan(cat_pos, mouse_pos)
            return 11 - dist - 0.1 * self.count
        else:
            if self.state[0] == self.state[1]:
                return 10
            elif (self.board[index2pos(self.state[0], self.board.shape)]
                == Label.block.value):
                return -10
            else:
                return -1
