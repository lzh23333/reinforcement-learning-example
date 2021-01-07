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
from enum import Enum
from utils.rl_base import Agent, Env


# define actions
ACTIONS = Enum("ACTIONS", ("up", "right", "down", "left"))


class CatAgent(Agent):
    """Cat Agent for example.

    Attributes:
        Q (np.ndarray): [n_states, 4].
        actions: []
    """
    def recv(self, state, r):
        pass
