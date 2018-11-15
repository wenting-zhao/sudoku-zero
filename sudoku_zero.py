import numpy as np
import random
import copy
import tensorflow as tf

import os
import sys
import argparse
import threading
import time
import importlib

from multiprocessing import Process, current_process, Value, Lock
from multiprocessing.queues import SimpleQueue
from multiprocessing.managers import BaseManager

from queue import Queue

from model import model as M
from mcts import MCTS

def whether_remove(prob):
    return (True if random.uniform(0, 1) < prob else False)

def meditation(random_state, gpu_id, queue, lock, verbose=True):
    cur_process = current_process()
    print ("Creat process pname=%s, pid=%d" % (cur_process.name, cur_process.pid))

    board_size = args.board_size 
    module = importlib.import_module(args.type)
    model = module.sudoku_model(args, mode="predict", gpu_list=str(gpu_id))
    #model = M(args, mode="predict", gpu_list=str(gpu_id))
    model.preprocess()
    model.build_model()

    game_in_thread = 0

    all_sudoku = np.load("datasets/complete_16.npy")

    while True:
        try:
            with lock:
                # Something you want to share through processes
                pass

            start_time = time.time()
            is_print = verbose and game_in_thread % 10 == 0

            # Renew the model based on some rules
            model.load_model()

            step = model.get_step()

            # Interface
            # mcts = MCTS()
            # TODO: MCTS Logic
            # You could pass the model as an argument into the MCTS class, and use model.predict to get the prediction given by NN
            mcts = MCTS(model=model, rollout=100, sudoku_size=16, ucb1_confidence=1, tree_policy="UCB1")
            sudoku = copy.deepcopy(all_sudoku[random.randrange(10000)])
            update_sudoku(sudoku)
            data = None
            for _ in range(100):
                res = mcts(sudoku, n=100000)
                data.append(res)

            # JUST FOR TEST: Random generate data for test
            #data = (np.random.random((9, 9, 10)), np.random.random(82), 10.0)
            # (sudoku, (x, y), action, distribution)
            # [([n, n], ((x, y), {1--n}, list))]
            queue.put(data)

        except Exception as e:
            print (str(e))

def update_sudoku(sudoku, prob=0.66):
    for (x, y), _ in np.ndenumerate(sudoku):
        if whether_remove(prob) is True:
            sudoku[x, y] = 0

def slave():
    BaseManager.register("get_history_queue")
    print ("Connect to server %s:%d..." % (args.host, args.port))
    manager = BaseManager(address=(args.host, args.port), authkey=bytes("miaomiaomiao", "utf-8"))
    manager.connect()
    queue = manager.get_history_queue()

    lock = Lock()

    gpu_ids = args.gpu_list.split(",")
    for i in range(args.num_thread):
        sub_process = Process(target=meditation, args=[np.random.RandomState(np.random.randint(0, 1000000)), i % len(gpu_ids), queue, lock, i == 0])
        sub_process.start()

    while True:
        pass

def train_worker(train_agent):
    summary_writter = tf.summary.FileWriter("%s-sudoku-zero.log" % args.type, train_agent.sess.graph)
    start_time = time.time()
    while True:
        global_step = train_agent.get_step()
        train_agent.train(summary_writter, print_step=20)

        if global_step % args.save_every == 0 and global_step > 0:
            train_agent.saver.save(train_agent.sess, os.path.join(args.model_path, "model.ckpt"), global_step=global_step)

def train(cluster):
    if cluster:
        queue = Queue(maxsize=128000)
        BaseManager.register("get_history_queue", callable=lambda: queue)
        manager = BaseManager(address=("0.0.0.0", args.port), authkey=bytes("miaomiaomiao", "utf-8"))
        manager.start()
        print ("Listen at %d..." % (args.port))

        queue = manager.get_history_queue()
    else:
        queue = SimpleQueue()

        lock = Lock()

        gpu_ids = args.gpu_list.split(",")
        for i in range(args.num_thread):
            sub_process = Process(target=meditation, args=[np.random.RandomState(np.random.randint(0, 1000000)), i % len(gpu_ids), queue, lock, i == 0])
            sub_process.start()

    module = importlib.import_module(args.type)
    train_agent = module.sudoku_model(args, mode="train", gpu_list=args.train_gpu)
    #train_agent = M(args, mode="train", gpu_list=args.train_gpu)
    train_agent.preprocess()
    train_agent.build_model()
    train_agent.load_model()
    
    train_thread = threading.Thread(target=train_worker, args=[train_agent])
    train_thread.daemon = True
    train_thread.start()
    
    n_sample = 0
    saved_data = 0
    total_data = 0
    variable_order_data = []
    while True:
        #(history, nxt_move, label) = queue.get()
        # Process the data
        historys = queue.get()
        #reward = float(historys[-1]) / 256.0
        for history in historys:
            reward = history[1]
            for i, item in enumerate(history[0]):
                nxt_move = np.zeros((args.board_size * args.board_size))
                nxt_move[(item[2][0] - 1) * args.board_size + item[2][1] - 1] = 1.0
                state = item[0]
                pos = []
                pos_value = []
                for j, candidata_pos in enumerate(item[1]):
                    pos.append(candidate_pos[0])
                    pos_value.append(candidate_pos[1])
                for _ in xrange(args.board_size * args.board_size - len(pos)):
                    pos.append((-1, -1))
                variable_order_data.append((np.array(state), np.array(nxt_move), reward, np.array(pos), np.array(pos_value)))
                if n_sample % 1000 == 0:
                    size = train_agent.push_sample(np.array(state), np.array(nxt_move), reward, np.array(pos), get_cur_size=True)
                    print ("Num training sample=%d, tf queue size=%d" % (n_sample, size))
                else:
                    train_agent.push_sample(np.array(state), np.array(nxt_move), reward, np.array(pos))
                n_sample += 1
                saved_data += 1
                if saved_data == int(1e7):
                    total_data += 1
                    saved_data = 0
                    np.save("./ining_data_variable_order/train_data_%d" % (total_data), variable_order_data)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--type", type=str)
    argparser.add_argument("--model_path", type=str, default=None)
    argparser.add_argument("--gpu_list", type=str, default="0")
    argparser.add_argument("--feature_num", type=int, default=18)
    argparser.add_argument("--mode", type=str, default="miao")
    argparser.add_argument("--load", type=int, default=0)
    argparser.add_argument("--lr", type=float, default=0.0001)
    argparser.add_argument("--l2", type=float, default=0.0001)

    argparser.add_argument("--port", type=int, default="12345")
    argparser.add_argument("--host", type=str, default="localhost")

    argparser.add_argument("--num_thread", type=int, default=1)
    argparser.add_argument("--eval_batch_size", type=int, default=16)

    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--buffer_size", type=int, default=1000000)
    argparser.add_argument("--min_buffer_size", type=int, default=10000)
    argparser.add_argument("--train_gpu", type=str, default='0')
    argparser.add_argument("--filename", type=str, default=None)
    argparser.add_argument("--regularizer", type=bool, default=True)
    argparser.add_argument("--board_size", type=int, default=16)
    argparser.add_argument("--save_every", type=int, default=1000)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

    print ("Args: ", args)
    if args.mode == "slave":
        slave()
    elif args.mode == "master":
        train(cluster=True)
    else:
        train(cluster=False)
    
