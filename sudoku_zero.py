import numpy as np
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

from Queue import Queue

def meditation(random_state, gpu_id, queue, lock, verbose=True):
    cur_process = current_process()
    print ("Creat process pname=%s, pid=%d" % (cur_process.name, cur_process.pid))

    board_size = args.board_size 
    module = importlib.import_moduel(args.type)
    model = module.sudoku_model(args, mode="predict", gpu_list=str(gpu_id))
    model.preprocess()
    model.build_model()

    game_in_thread = 0
    
    while True:
        try:
            with lock:
                # Something you want to share through processes
                pass

            start_time = time.time()
            is_print = verbose and game_in_thread % 10 == 0
            model.load_model()

            step = model.get_step()

            # Interface
            #mcts = MCTS()
            # TODO: MCTS Logic

            # TODO: Collect the data

            # JUST FOR TEST: Random generate data for test
            data = (np.random.random((9, 9, 10)), 2, 10)
            queue.put(data)
            
        except Exception, e:
            print (str(e))

def slave():
    BaseManager.register("get_history_queue")
    print ("Connect to server %s:%d..." % (args.host, args.port))
    manager = BaseManager(address=(args.host, args.port), authkey="miaomiaomiao")
    manager.connect()
    queue = manager.get_history_queue()

    lock = Lock()

    gpu_ids = args.gpu_list.aplit(",")
    for i in range(args.num_thread):
        sub_process = Process(target=meditation, args=[np.random.RandomState(np.random.randint(0, 1000000)), i % len(gpu_ids), queue, lock, i == 0])
        sub_process.start()

    while True:
        pass

def train_worker(train_agent):
    summary_writer = tf.summary.FileWiter("%s-sudoku-zero.log" % args.type, train_agent.sess.graph)
    start_time = time.time()
    while True:
        global_step = train_agent.get_step()
        train_agent.train(summary_writter, n_step=20)

        if global_step % args.save_every == 0 and global_step > 0:
            train_agent.saver.save(train_agent.sess, os.path.join(args.model_path, "model.ckpt"), global_step=global_step)

def train(cluster):
    if cluster:
        queue = Queue(maxsize=128000)
        BaseManager.register("get_history_queue", callable=lambda: queue)
        manager = BaseManager(address=("0.0.0.0", args.port), authkey="miaomiaomiao")
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
    train_agent.preprocess()
    train_agent.build_model()
    train_agent.load_model()
    
    train_thread = threading.Thread(target=train_worker, args=[train_agent])
    train_thread.daemon = True
    train_thread.start()
    
    n_sample = 0
    while True:
        (history, nxt_move, label) = queue.get()
        if n_sample % 1000 == 0:
            size = train_agent.push_sample(history, nxt_move, label, ret_size=True)
            print ("Num training sample=%d, tf queue size=%d" % (n_sample, size))
        else:
            train_agent.push_sample(history, nxt_move, label)
        n_sample += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--type", type=str)
    argparser.add_argument("--model_path", type=str, default=None)
    argparser.add_argument("--gpu_list", type=str, default="0")
    argparser.add_argument("--num_feature", type=int, default=9*9*10)

    argparser.add_argument("--port", type=int, default="12345")
    argparser.add_argument("--host", type=str, default="localhost")

    argparser.add_argument("--num_thread", type=int, default=1)
    argparser.add_argument("--eval_batch_size", type=int, default=16)

    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--train_gpu", type=str, default='0')
    argparser.add_argument("--filename", type=str, default=None)
    argparser.add_argument("--regularizer", type=bool, default=True)
    argparser.add_argument("--board_size", type=int, default=9)
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
    
