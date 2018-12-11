from mcts import MCTS
import random
import numpy as np
import sys
import copy
import importlib
import tensorflow as tf
import argparse
import os

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

sudoku = np.asarray([[ 5,  0,  0, 13,  2,  0,  0, 14,  0,  0,  0, 15,  4,  0,  0,  0],
                    [ 2,  6, 10,  0,  1,  5,  0,  0,  0,  8,  0,  0,  0,  0,  0, 15],
                    [ 3,  0,  0,  0,  0,  0,  0, 16,  0,  0,  0,  0,  0,  0, 10, 14],
                    [ 4,  8,  0,  0,  0,  0,  0,  0,  0,  6, 10,  0,  0,  5,  0,  0],
                    [ 0,  5,  0,  0,  0,  2, 14,  0,  0,  0, 15,  0,  0,  0, 16,  0],
                    [ 6,  2,  0,  0,  0,  0, 13,  0,  0,  4, 16,  0,  0,  3, 15,  0],
                    [ 7,  0,  0, 11,  0,  0,  0,  0,  5,  1,  0,  9,  6,  0, 14, 10],
                    [ 0,  0,  0, 12,  0,  3,  0,  0,  6,  0,  0,  0,  5,  1, 13,  0],
                    [ 0,  0,  1,  0, 10,  0,  0,  0, 11,  0,  0,  7, 12,  0,  4,  0],
                    [ 0,  0,  2,  6,  9, 13,  1,  0,  0,  0,  0,  0, 11, 15,  0,  0],
                    [11,  0,  0,  0, 12,  0,  4,  0,  9,  0,  0,  5,  0,  0,  0,  0],
                    [ 0, 16,  0,  0,  0, 15,  3,  0,  0,  0,  2,  0,  0,  0,  1,  0],
                    [13,  0,  5,  1,  0,  0,  0,  2, 15, 11,  7,  3,  0, 12,  0,  0],
                    [ 0,  0,  0,  0, 13,  0,  0,  0, 16,  0,  0,  0,  0, 11,  0,  0],
                    [ 0,  0,  7,  0, 16,  0,  0,  0,  0,  0,  5,  0, 14,  0,  6,  0],
                    [ 0, 12,  0,  0,  0, 11,  7,  3,  0,  0,  0,  2,  0,  0,  0,  0]])

# sudoku = np.zeros((16, 16))

# all_sudoku = np.load("datasets/complete_16.npy")
# sudoku = all_sudoku[random.randrange(10000)]
orig = copy.deepcopy(sudoku)

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0


def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)


def main():
    module = importlib.import_module(args.type)
    train_agent = module.sudoku_model(args, mode="predict", gpu_list=args.train_gpu)
    train_agent.sl_preprocess()
    train_agent.sl_build_model()
    train_agent.load_model()

    mcts = MCTS(train_agent, sudoku_size=16, infer=True, rollout=100, ucb1_confidence=0)
    mcts.set_least_val_first()
    res = mcts(sudoku, n=100000)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--type", type=str)
    argparser.add_argument("--model_path", type=str, default=None)
    argparser.add_argument("--gpu_list", type=str, default="0")
    argparser.add_argument("--feature_num", type=int, default=18)
    argparser.add_argument("--load", type=int, default=0)
    argparser.add_argument("--lr", type=float, default=0.0001)
    argparser.add_argument("--l2", type=float, default=0.0001)

    argparser.add_argument("--port", type=int, default="12345")
    argparser.add_argument("--host", type=str, default="localhost")
    argparser.add_argument("--log", type=str, default="default_log")

    argparser.add_argument("--num_thread", type=int, default=1)
    argparser.add_argument("--eval_batch_size", type=int, default=16)

    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--num_data", type=int, default=128)
    argparser.add_argument("--buffer_size", type=int, default=1000000)
    argparser.add_argument("--min_buffer_size", type=int, default=10000)
    argparser.add_argument("--train_gpu", type=str, default='0')
    argparser.add_argument("--filename", type=str, default=None)
    argparser.add_argument("--regularizer", type=bool, default=True)
    argparser.add_argument("--board_size", type=int, default=16)
    argparser.add_argument("--save_every", type=int, default=1000)
    # 1: Policy 2: Value 3: P+V
    argparser.add_argument("--model_type", type=int, default=1)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

    print ("Args: ", args)
    main()
