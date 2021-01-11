#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/01/11 09:37:39
@Author  :   lzh 
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   q learning and store Q, board.
'''
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from controller import Controller


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--board_size", type=int, nargs=2, default=(10, 10))
    parser.add_argument("--blocks", type=int,
                        help="num of blocks (random pos)", default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=0.2, help="eps-greedy")
    parser.add_argument("--eta", type=float, default=0.7)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--mouse_pattern", type=str,
                        choices=["stay", "random"])
    parser.add_argument("--dst", type=str,
                        help="controller store path", default="c.pkl")
    return parser.parse_args()


def main():
    # set params
    args = parse_args()
    board_size = args.board_size
    init_state = (0, board_size[0] * board_size[1] - 1)
    mouse_pattern = args.mouse_pattern
    eps = args.eps
    lr = args.lr
    eta = args.eta
    max_iter = args.max_iter
    blocks = args.blocks

    #
    c = Controller(
        board_size,
        init_state,
        block_num=blocks,
        eps=eps,
        mouse_move=mouse_pattern
    )

    # q learning
    rewards = c.q_learning(lr=lr, eta=eta, max_iter=max_iter)

    with open(args.dst, "wb") as f:
        pickle.dump(c, f)

    # plot reward-episode curve
    plt.figure(2)
    plt.plot(range(max_iter), rewards, linewidth=1)
    plt.xlim([0, max_iter])
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(f"q learning, lr={lr}, eta={eta}")
    plt.grid("on")
    plt.show()


if __name__ == "__main__":
    main()
