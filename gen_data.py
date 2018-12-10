import numpy as np
import pickle
import sys
import copy
import random
import string
from statistics import mean

from mcts import MCTS


def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0

def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def main():
    n = int(sys.argv[1])
    all_sudoku = np.load("datasets/train_16.npy")
    for i in range(n):
        mcts = MCTS(model=None, rollout=100, sudoku_size=16, gen_data=True, ucb1_confidence=1)
        sudoku = copy.deepcopy(all_sudoku[random.randrange(10000)])
        update_sudoku(sudoku)
        idx = 0
        data = []
        rollouts = []
        while idx < 10:
            res = mcts(sudoku, n=1000)
            rollouts.append(res[-1])
            if res is None:
                continue
            else:
                data.append(res)
                idx += 1
        if mean(rollouts) < 10:
            continue
        pickle.dump(data, open("new_{}.p".format(id_generator()), "wb"))


if __name__ == '__main__':
    main()

