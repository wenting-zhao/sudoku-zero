import numpy as np
import pickle
import sys
import copy
import random

from mcts import MCTS


def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0

def main():
    n = int(sys.argv[1])
    all_sudoku = np.load("datasets/complete_16.npy")
    for i in range(n):
        mcts = MCTS(model=None, rollout=100, sudoku_size=16, ucb1_confidence=1, tree_policy="UCB1")
        sudoku = copy.deepcopy(all_sudoku[random.randrange(10000)])
        update_sudoku(sudoku)
        idx = 0
        data = []
        while idx < 100:
            res = mcts(sudoku, n=1000)
            print(res)
            if res is None:
                continue
            else:
                data.append(res)
                idx += 1
        pickle.dump(data, open("{}.p".format(i), "wb"))


if __name__ == '__main__':
    main()