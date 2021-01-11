#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2021/01/11 10:11:47
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   show train result
'''
import pickle
from argparse import ArgumentParser
from gui import BoardGUI


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("filename", type=str, help="controller pkl")
    parser.add_argument("--ms", type=float, help="animation interval", default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.filename, "rb") as f:
        c = pickle.load(f)

    state_history = c.epsiode()
    BoardGUI(c.board, c.init_state, state_history, ms=args.ms)


if __name__ == "__main__":
    main()
