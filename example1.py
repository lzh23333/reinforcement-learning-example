# %%
from argparse import ArgumentParser, ArgumentTypeError
import matplotlib.pyplot as plt
from controller import Controller
from gui import BoardGUI


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--board_size', help="size of board", type=int, nargs=2, default=(40, 40))
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=0.2, help="eps-greedy")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=5000)
    parser.add_argument("--mouse_pattern", type=str, choices=["stay", "random"])
    parser.add_argument("--ms", type=float, help="animation interval", default=0.5)
    return parser.parse_args()


def main():
    # set params
    args = parse_args()
    board_size = args.board_size
    print(board_size)
    init_state = (0, board_size[0] * board_size[1] - 1)
    mouse_pattern = args.mouse_pattern
    eps = args.eps
    lr = args.lr
    eta = args.eta
    max_iter = args.max_iter

    # 
    c = Controller(
        board_size,
        init_state,
        eps=eps,
        mouse_move=mouse_pattern
    )

    # %%
    rewards = c.q_learning(lr=lr, eta=eta, max_iter=max_iter)
    # %%

    # get optimal actions and history
    state_history = c.epsiode()
    print(state_history)

    # GUI
    BoardGUI(c.board, c.init_state, state_history, args.ms)

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