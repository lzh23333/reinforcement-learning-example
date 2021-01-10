import numpy as np
from gui import BoardGUI


init_state = (0, 15)
state_history = [(0, 15), (1, 15), (2, 15), (3, 15), (7, 15), (11, 15), (15, 15)]
board = np.zeros((4, 4))
board[[1, 2], [2, 1]] = 1

g = BoardGUI(board, init_state, state_history)
g.run()
