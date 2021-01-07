#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rl.py
@Time    :   2021/01/07 16:27:53
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   reinforcement learning utils
'''


class Env(object):
    """Base Class for Environment in reinforcement learning.

    Attributes:
        recv: recieve action from agent, and make changes to env.
        response: get latest state S_{t+1} and reward R_{t}.
        states: current states.

    """

    def __init__(self, state, reward):
        self.state = state
        self.reward = reward

    @property
    def state(self):
        return self.state

    def recv(self, action):
        pass

    def response(self):
        return self.state, self.reward(self.state)


class Agent(object):
    """Base Class for Agent in RL.

    Attributes:
        recv: revieve env's state and reward.
        action: take an action.
        actions: action set.
    """

    def __init__(self, Q, actions):
        self.Q = Q
        self.S = None
        self.actions = actions

    @property
    def Q(self):
        return self.Q

    def recv(self, state, r):
        pass

    def action(self):
        pass

