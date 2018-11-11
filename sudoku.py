import mcts as m1
import another_mcts as m2
import random
import numpy as np
import sys
import copy

# sudoku = np.asarray([[0, 0, 0, 0, 0, 0, 0, 1, 0],
#                      [4, 0, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 2, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 0, 5, 0, 4, 0, 7],
#                      [0, 0, 8, 0, 0, 0, 3, 0, 0],
#                      [0, 0, 1, 0, 9, 0, 0, 0, 0],
#                      [3, 0, 0, 4, 0, 0, 2, 0, 0],
#                      [0, 5, 0, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 0, 8, 0, 6, 0, 0, 0]])

# sudoku = np.asarray([[ 5,  1,  9, 13,  2,  6, 10, 14,  3,  7, 11, 15,  4,  8, 12, 16],
#                      [ 2,  6, 10, 14,  1,  5,  9, 13,  4,  8, 12, 16,  3,  7, 11, 15],
#                      [ 3,  7, 11, 15,  4,  8, 12, 16,  1,  5,  9, 13,  2,  6, 10, 14],
#                      [ 4,  8, 12, 16,  3,  7, 11, 15,  2,  6, 10, 14,  1,  5,  9, 13],
#                      [ 1,  5, 13,  9,  6,  2, 14, 10,  7,  3, 15, 11,  8,  4, 16, 12],
#                      [ 6,  2, 14, 10,  5,  1, 13,  9,  8,  4, 16, 12,  7,  3, 15, 11],
#                      [ 7,  3, 15, 11,  8,  4, 16, 12,  5,  1, 13,  9,  6,  2, 14, 10],
#                      [ 8,  4, 16, 12,  7,  3, 15, 11,  6,  2, 14, 10,  5,  1, 13,  9],
#                      [ 9, 13,  1,  5, 10, 14,  2,  6, 11, 15,  3,  7, 12, 16,  4,  8],
#                      [10, 14,  2,  6,  9, 13,  1,  5, 12, 16,  4,  8, 11, 15,  3,  7],
#                      [11, 15,  3,  7, 12, 16,  4,  8,  9, 13,  1,  5, 10, 14,  2,  6],
#                      [12, 16,  4,  8, 11, 15,  3,  7, 10, 14,  2,  6,  9, 13,  1,  5],
#                      [13,  9,  5,  1, 14, 10,  6,  2, 15, 11,  7,  3, 16, 12,  8,  4],
#                      [14, 10,  6,  2, 13,  9,  5,  1, 16, 12,  8,  4, 15, 11,  7,  3],
#                      [15, 11,  7,  3, 16, 12,  8,  4, 13,  9,  5,  1, 14, 10,  6,  2],
#                      [16, 12,  8,  4, 15, 11,  7,  3, 14, 10,  6,  2, 13,  9,  5,  1]])

# sudoku = np.asarray([[ 5,  0,  0, 13,  2,  0,  0, 14,  0,  0,  0, 15,  4,  0,  0,  0],
#                     [ 2,  6, 10,  0,  1,  5,  0,  0,  0,  8,  0,  0,  0,  0,  0, 15],
#                     [ 3,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0,  0, 10, 14],
#                     [ 4,  8,  0,  0,  0,  0,  0,  0,  0,  6, 10,  0,  0,  5,  0,  0],
#                     [ 0,  5,  0,  0,  0,  2, 14,  0,  0,  0, 15,  0,  0,  0, 16,  0],
#                     [ 6,  2,  0,  0,  0,  0, 13,  0,  0,  4, 16,  0,  0,  3, 15,  0],
#                     [ 7,  0,  0, 11,  0,  0,  0,  0,  5,  1,  0,  9,  6,  0, 14, 10],
#                     [ 0,  0,  0, 12,  0,  3,  0,  0,  6,  0,  0,  0,  5,  1, 13,  0],
#                     [ 0,  0,  1,  0, 10,  0,  0,  0, 11,  0,  0,  7, 12,  0,  4,  0],
#                     [ 0,  0,  2,  6,  9, 13,  1,  0,  0,  0,  0,  0, 11, 15,  0,  0],
#                     [11,  0,  0,  0, 12,  0,  4,  0,  9,  0,  0,  5,  0,  0,  0,  0],
#                     [ 0, 16,  0,  0,  0, 15,  3,  0,  0,  0,  2,  0,  0,  0,  1,  0],
#                     [13,  0,  5,  1,  0,  0,  0,  2, 15, 11,  7,  3,  0, 12,  0,  0],
#                     [ 0,  0,  0,  0, 13,  0,  0,  0, 16,  0,  0,  0,  0, 11,  0,  0],
#                     [ 0,  0,  7,  0, 16,  0,  0,  0,  0,  0,  5,  0, 14,  0,  6,  0],
#                     [ 0, 12,  0,  0,  0, 11,  7,  3,  0,  0,  0,  2,  0,  0,  0,  0]])

all_sudoku = np.load("datasets/complete_25.npy")
sudoku = all_sudoku[random.randrange(10000)]
orig = copy.deepcopy(sudoku)

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0


def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)


def main():
    n = int(sys.argv[1])
    c_map = {9: 5, 16: 16, 25: 38}
    #sudoku = np.zeros((n, n))
    update_sudoku(sudoku)
    # mcts = m1.MCTS(None, sudoku_size=n, ucb1_confidence=c_map[n], tree_policy="UCB1")
    # while 0 in sudoku[:, :]:
    #     res = mcts(sudoku, n=100)
    #     if res[0][1] == "unsatisfiable":
    #         print("unsatisfiable", (x, y))
    #         print(sudoku)
    #         print(np.count_nonzero(sudoku))
    #         print(orig)
    #         break

    #     # since a solution can be found during rollout,
    #     # res can be more than one best action.
    #     for one in res[:-1]:
    #         (x, y), action = one[:2]
    #         sudoku[x, y] = action
    #         print(one[-1])
    #     if len(res) > 1:
    #         break
    # print(sudoku)

    mcts = m2.MCTS(None, sudoku_size=n, ucb1_confidence=5, tree_policy="UCB1")
    for res in mcts(sudoku, n=100000):
        print(res)
        if res[-1] == "yes":
            print("result found.")
            break
        elif res[-1] == "no":
            print("no solution found.")

if __name__ == '__main__':
    main()
