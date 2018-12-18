import numpy as np
import random
import copy
import tensorflow as tf

import os
import sys
import argparse
import time
import importlib

def train():
    module = importlib.import_module(args.type)
    train_agent = module.sudoku_model(args, mode="train", gpu_list=args.train_gpu)
    train_agent.sl_preprocess(args.model_type)
    train_agent.sl_build_model(args.model_type)
    train_agent.load_model()

    summary_writter = tf.summary.FileWriter("%s-modeltype_%d-sl.log" % (args.log, args.model_type), train_agent.sess.graph)    
    start_time = time.time()
    train_data = np.load("final_sudoku_sl_data_train.npy")
    train_label = np.load("final_sudoku_sl_label.npy")
    train_value = None
    if args.model_type != 1:
        train_value = np.load("final_sudoku_sl_value.npy")
    shuffle_index = np.arange(train_data.shape[0])
    np.random.shuffle(shuffle_index)
    train_data = train_data[shuffle_index]
    train_label = train_label[shuffle_index]
    if args.model_type != 1:
        train_value = train_value[shuffle_index]
    num_data = train_data.shape[0]
    print ("Total training data number: %d" % num_data)
    for iteration in range(3):
        print ("Training on iteration: %d" % (iteration))
        for i in range(int(num_data // args.batch_size)):
            if (i + 1) * args.batch_size > num_data:
                break
            print ("Training on %d-th minibatch" % i)
            st = i * args.batch_size
            ed = (i + 1) * args.batch_size
            X = train_data[st:ed]
            label = train_label[st:ed]
            if args.model_type != 1:
                value = train_value[st:ed]
            global_step = train_agent.get_step()
            if args.model_type != 1:
                train_agent.sl_train(summary_writter, features=np.array(X), labels=np.array(label), values=np.array(value), model_type=args.model_type, print_step=10)
            else:
                train_agent.sl_train(summary_writter, features=np.array(X), labels=np.array(label), values=None, model_type=args.model_type, print_step=10)

            if global_step % args.save_every == 0 and global_step > 0:
                train_agent.saver.save(train_agent.sess, os.path.join(args.model_path, "model.ckpt"), global_step=global_step)

    global_step = train_agent.get_step()
    train_agent.saver.save(train_agent.sess, os.path.join(args.model_path, "model.ckpt"), global_step=global_step)

if __name__ == "__main__":
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
    argparser.add_argument("--save_every", type=int, default=500000)
    # 1: Policy 2: Value 3: P+V
    argparser.add_argument("--model_type", type=int, default=1)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list

    print ("Args: ", args)
    train()
 

