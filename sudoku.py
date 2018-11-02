from mcts import MCTS
import numpy as np
import sys

# sudoku = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0],
#                      [4, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 2, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 5, 0, 4, 0, 7],
#                      [0, 0, 8, 0, 0, 0, 3, 0, 0],
#                      [0, 0, 1, 0, 9, 0, 0, 0, 0],
#                      [3, 0, 0, 4, 0, 0, 2, 0, 0],
#                      [0, 5, 0, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 8, 0, 6, 0, 0, 0]])


def main():
    n = int(sys.argv[1])
    c_map = {9: 5, 16: 16}
    sudoku = np.zeros((n, n))
    mcts = MCTS(sudoku_size=n, ucb1_confidence=c_map[n], tree_policy="UCB1")
    while 0 in sudoku[:, :]:
        res = mcts(sudoku, n=100)
        # since a solution can be found during rollout,
        # res can be more than one best action.
        for move in res:
            (x, y), action = move
        if len(res) > 1:
            break
        if action == "unsatisfiable":
            print("unsatisfiable", (x, y))
            print(sudoku)
            print(np.count_nonzero(sudoku))
            break
        sudoku[x, y] = action
        #print("best_action", (x, y), action)
        #print(sudoku)

if __name__ == '__main__':
    main()
