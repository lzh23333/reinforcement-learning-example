#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/01/08 15:36:59
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   utils function
'''
from enum import Enum


# define actions
class Move(Enum):
    up = 0
    right = 1
    down = 2
    left = 3


# define cat, block, empty, mouse
class Label(Enum):
    empty = 0
    block = 1
    mouse = 2
    cat = 3


def pos_assert(pos, board_size):
    assert pos[0] < board_size[0] and pos[1] < board_size[1] and\
        pos[0] >= 0 and pos[1] >= 0,\
        "pos out of range"


def action_assert(action):
    assert action in range(4), "action value should in [0, 1, 2 3]"


def pos2index(pos, board_size):
    """transform (x, y) to index.
    """
    pos_assert(pos, board_size)
    return board_size[0] * pos[0] + pos[1]


def index2pos(index, board_size):
    assert index >= 0 and index < board_size[0] * board_size[1] - 1,\
        "index out of range"
    return (index // board_size[0], index % board_size[0])


def move_on_board(pos, action, board_size):
    """

    Args:
        pos (tuple (int, int)): (x, y), start from (0, 0).
        action (int): Move.
        board_size (tuple (int, int)): (X, Y).
    Returns:
        new_pos (tuple (int, int)): (x', y').
    """
    action_assert(action)
    pos_assert(pos, board_size)
    new_pos = list(pos)
    if Move(action) == Move.up:
        new_pos[0] = max(0, new_pos[0] - 1)
    elif Move(action) == Move.down:
        new_pos[0] = min(board_size[0] - 1, new_pos[0] + 1)
    elif Move(action) == Move.left:
        new_pos[1] = max(0, new_pos[1] - 1)
    else:
        new_pos[1] = min(board_size[1] - 1, new_pos[1] - 1)
    return new_pos
