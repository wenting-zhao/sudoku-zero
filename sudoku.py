from mcts import MCTS
import numpy as np

sudoku = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [4, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 2, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 5, 0, 4, 0, 7],
                     [0, 0, 8, 0, 0, 0, 3, 0, 0],
                     [0, 0, 1, 0, 9, 0, 0, 0, 0],
                     [3, 0, 0, 4, 0, 0, 2, 0, 0],
                     [0, 5, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 8, 0, 6, 0, 0, 0]])


def main():
    mcts = MCTS(sudoku_size=9)
    while 0 in sudoku[:, :]:
        (x, y), action = mcts(sudoku)
        sudoku[x, y] = action
        print("best_action", (x, y), action)

if __name__ == '__main__':
    main()
