import argparse
import numpy as np
import sys
import importlib
import tensorflow as tf
import random

from mcts import MCTS
from baselines.nmct.nested import NestedMCTS
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, default=10,
                        help="the number of times running the instance")
    parser.add_argument("--instances", type=int, default=100,
                        help="the number of times running the instance")
    parser.add_argument('--rollout', action='store_true', default=False,
                           help="least number heuristic in rollout")
    parser.add_argument('--next_node', action='store_true', default=False,
                           help="least number heuristic in get_next_node")

    parser.add_argument("--type", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--gpu_list", type=str, default="0")
    parser.add_argument("--feature_num", type=int, default=18)
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=0.0001)

    parser.add_argument("--port", type=int, default="12345")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--log", type=str, default="default_log")

    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_data", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=1000000)
    parser.add_argument("--min_buffer_size", type=int, default=10000)
    parser.add_argument("--train_gpu", type=str, default='0')
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--regularizer", type=bool, default=True)
    parser.add_argument("--board_size", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=1000)
    # 1: Policy 2: Value 3: P+V
    parser.add_argument("--model_type", type=int, default=1)
    parser.add_argument("--model_index", type=str, default=None)
    args = parser.parse_args()

    return args

def print_result(stats):
    # print stats
    times = stats.get_times()
    counts = stats.get_counts()
    other = stats.get_stats()

    # sort categories by total runtime
    categories = sorted(times, key=times.get)
    maxlen = max(len(x) for x in categories)
    for category in categories:
        sys.stdout.write("%-*s : %8.3f\n" % (maxlen, category, times[category]))
    for category in sorted(counts):
        sys.stdout.write("%-*s : %8d\n" % (maxlen + 6, category + ' count', counts[category]))
        if category in times:
            sys.stdout.write("%-*s : %8.5f\n" % (maxlen + 6, category + ' per', times[category] / counts[category]))

    # print min, max, avg of other values recorded
    if other:
        maxlen = max(len(x) for x in other)
        for name, values in other.items():
            sys.stdout.write("%-*s : %f\n" % (maxlen + 4, name + ' min', min(values)))
            sys.stdout.write("%-*s : %f\n" % (maxlen + 4, name + ' max', max(values)))
            sys.stdout.write("%-*s : %f\n" % (maxlen + 4, name + ' avg', np.mean(values)))
            sys.stdout.write("%-*s : %f\n" % (maxlen + 4, name + ' median', np.median(values)))

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0

def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)

def main():
    args = parse_args()
    module = importlib.import_module(args.type)
    train_agent = module.sudoku_model(args, mode="predict", gpu_list=args.train_gpu)
    train_agent.sl_preprocess(args.model_type)
    train_agent.sl_build_model(args.model_type)
    train_agent.load_model(index=args.model_index)

    for i in range(args.instances):
        stats = utils.Statistics()
        all_sudoku = np.load("datasets/train_16.npy")
        #sudoku = all_sudoku[random.randrange(10000)]
        sudoku = all_sudoku[0]
        update_sudoku(sudoku)
        n = len(sudoku)

        mcts = MCTS(None, sudoku_size=n, rollout=100, ucb1_confidence=0)
        mcts.set_least_val_first(rollout=args.rollout, next_node=args.next_node)
        for _ in range(args.number):
            with stats.time('mcts'):
                res = mcts(sudoku, n=100000)
                if res is not None:
                    stats.add_stat('mcts_counter', mcts.sample)
                else:
                    stats.add_stat('mcts_counter', 999999)
                mcts.reset_sample()

        dlmcts = MCTS(train_agent, sudoku_size=16, infer=True, rollout=100, ucb1_confidence=0, softmax=True)
        dlmcts.set_least_val_first(rollout=args.rollout, next_node=args.next_node)
        for _ in range(args.number):
            with stats.time('dlmcts'):
                res = dlmcts(sudoku, n=100000)
                if res is not None:
                    stats.add_stat('dlmcts_counter', dlmcts.sample)
                else:
                    stats.add_stat('dlmcts_counter', 999999)
                dlmcts.reset_sample()
        print("instance", i)
        print_result(stats)

if __name__ == '__main__':
    main()
