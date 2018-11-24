import numpy as np
import tensorflow as tf
import argparse
import time
import sys
import copy

from sklearn.metrics import accuracy_score
from model_base import model_base

def _placeholder(dtype=tf.float32, shape=None, name=None):
    return tf.placeholder(dtype=dtype, shape=shape, name=name)

class model(model_base):
    def __init__(self, args, mode, gpu_list=None):
        super().__init__(args, gpu_list)
        self.mode = mode
        self.batch_size = args.batch_size if self.mode == "train" else args.eval_batch_size
        self.gpu_list = [item for item in gpu_list.split(',')]
        self.args = args
        self.overall_acc = -1

    def _loss(self, logit, value, nxt_move, label, prob, regularizer):
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=nxt_move)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=nxt_move, logits=logit)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        #v_diff = tf.squared_difference(value, label)
        #v_loss = tf.reduce_mean(v_diff)

        reg_variables_gradients = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.add_n(reg_variables_gradients)

        eps = 1e-6
        kl_divergence = tf.reduce_sum(nxt_move * tf.log((nxt_move + eps) / (prob + eps)), axis=1)
        kl_mean = tf.reduce_mean(kl_divergence)

        #return cross_entropy_mean, v_loss, reg_term, cross_entropy_mean + reg_term, kl_mean
        return cross_entropy_mean, 0, reg_term, cross_entropy_mean + reg_term, kl_mean

    #def _extract_feature(self, history, pos):
    #    # TODO:
    #    X, Y = pos[0], pos[1]
    #    n_board = history.shape[0]
    #    ret = np.zeros((n_board, n_board, n_board + 2))
    #    for (x, y) in zip(range(n_board), range(n_board)):
    #        ret[x, y, history[x, y]] = 1.0
    #    ret[X, Y, n_board + 1] = 1.0
    #    return ret

    def _extract_feature(self, history, pos):
        print (history)
        print (history.shape)
        n_board = history.shape[0]
        ret = np.zeros((n_board, n_board, n_board + 2))
        print (ret.shape)
        for (x, y) in zip(range(n_board), range(n_board)):
            ret[x, y, int(history[x, y])] = 1.0
        for (x, y) in pos:
            if x == -1 and y == -1:
                break
            ret[x, y, n_board + 1] = 1.0
        return ret

    # For supervised training
    def sl_preprocess(self):
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='steps', trainable=False)
            if self.mode == "train":
                nxt_move = self.nxt_move = _placeholder(shape=[None, self.args.board_size * self.args.board_size], name="nxt_move")
                features = self.features = _placeholder(shape=[None, self.args.board_size, self.args.board_size, self.args.feature_num], name="features")
            else:
                if self.mode == "predict":
                    self.X = _placeholder(shape=[None, self.args.board_size, self.args.board_size, self.args.feature_num], name="infer_features")

    def sl_build_model(self):
        board_size = self.args.board_size
        batch_size = self.args.batch_size
        with self.graph.as_default():
            if self.mode == "predict":
                with tf.device('/gpu:%s' % self.gpu_list[0]):
                    logit, _ = self._forward(self.X, batch_size, False, dev='tower0', reuse=None)
                    prob = tf.nn.softmax(logit)
                    self.prob = tf.identity(prob, "policy_output")
                    #self.value = tf.identity(value, "value_output")
            else:
                #TODO:
                lr = tf.train.piecewise_constant(self.global_step, [500000, 1000000], [0.0005, 0.0001, 0.00001])
                #lr = self.args.lr

                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

                #X, nxt_move, label, pos = self.sample_buffer.dequeue_many(batch_size)
                n_gpu = len(self.gpu_list)
                #batch_X  = tf.split(X, n_gpu)
                batch_X  = tf.split(self.features, n_gpu)
                #batch_nm = tf.split(nxt_move, n_gpu)
                batch_nm = tf.split(self.nxt_move, n_gpu)
                batch_label = [0 for i in range(len(batch_nm))]

                tower_grads = []
                tower_loss  = []
                tower_prob  = []
                tower_ce    = []
                tower_mse   = []
                tower_reg   = []
                self.prob_acc = None

                for i, cur_gpu in enumerate(self.gpu_list):
                    cur_gpu = int(cur_gpu)
                    print ("Build graph on GPU: ", cur_gpu)
                    with tf.device('/device:GPU:%d' % cur_gpu):
                        with tf.name_scope('tower_%d' % cur_gpu) as scope:
                            gpu_batch_size = batch_size / n_gpu # This should be diveded 
                            regularizer = tf.contrib.layers.l2_regularizer(self.args.l2)
                            if i == 0:
                                #logit, prob, v = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=None, regularizer=regularizer)
                                logit, prob = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=None, regularizer=regularizer)
                            else:
                                #logit, prob, v = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=True, regularizer=regularizer)
                                logit, prob = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=True, regularizer=regularizer)

                            tower_prob.append(prob)
                            #ce, mse, reg, loss, kl = self._loss(logit, v, batch_nm[i], batch_label[i], prob, regularizer=regularizer)
                            ce, mse, reg, loss, kl = self._loss(logit, 0, batch_nm[i], batch_label[i], prob, regularizer=regularizer)
                            #acc = tf.metrics.accuracy(labels=tf.argmax(batch_nm[i]), predictions=tf.argmax(prob))

                            grads = optimizer.compute_gradients(loss)
                            if grads is None:
                                print (i)
                            tower_grads.append(grads)
                            tower_loss.append(loss)
                            tower_ce.append(ce)
                            tower_mse.append(mse)
                            tower_reg.append(reg)

                            tf.summary.scalar('/gpu:%d/mse'%i, mse)
                            tf.summary.scalar('/gpu:%d/ce'%i,ce)
                            tf.summary.scalar('/gpu:%d/reg'%i,reg)
                            tf.summary.scalar('/gpu:%d/loss'%i,loss)
                            tf.summary.scalar('/gpu:%d/kl'%i,kl)

                self.prob = tf.reshape(tower_prob, [batch_size, board_size * board_size])
                grads = self._average_gradients(tower_grads)
                apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)
                self.train_step = apply_gradient_op
                self.loss = tf.reduce_mean(tower_loss)
                self.ce   = tf.reduce_mean(tower_ce)
                self.mse  = tf.reduce_mean(tower_mse)
                self.reg  = tf.reduce_mean(reg)
                #self.acc = tf.metrics.accuracy(labels=tf.argmax(self.nxt_move), predictions=tf.argmax(self.prob))

                tf.summary.scalar("lr", lr)
                self.summary_step = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            self.all_var = tf.global_variables()



    def preprocess(self):
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='steps', trainable=False)
            if self.mode == "train":
                #history = self.history = _placeholder(tf.string, name="historys")
                nxt_move = self.nxt_move = _placeholder(shape=[self.args.board_size * self.args.board_size], name="nxt_move")
                label = self.label = _placeholder(name="label")
                # TODO: write extract feature as a tfop 
                #features, search_prob = self._extract_feature(history)
                #features, search_prob = history, history
                features = self.features = _placeholder(shape=[self.args.board_size, self.args.board_size, self.args.feature_num], name="features")
                pos = self.pos = _placeholder(shape=[None, 2], dtype=np.int32, name="input_position")
                # TODO: remove search prob
                search_prob = features

                self.sample_buffer = sample_buffer = tf.RandomShuffleQueue(capacity=self.args.buffer_size, min_after_dequeue=self.args.min_buffer_size, 
                                                                            shapes=[[self.args.board_size, self.args.board_size, self.args.feature_num], [self.args.board_size * self.args.board_size], [], [self.args.board_size * self.args.board_size, 2]], 
                                                                            dtypes=[tf.float32, tf.float32, tf.float32, tf.int32])

                self.feed_step = sample_buffer.enqueue((features, nxt_move, label, pos))
                self.queue_size = sample_buffer.size()
            else:
                if self.mode == "predict":
                    X = []
                    history = self.history = _placeholder(tf.string, name='history')
                    for i in range(self.batch_size):
                        # TODO: write extract feature as a tfop 
                        #feature, search_prob = _extract_feature(history[i])
                        features = history
                        X.append(features)
                    X = tf.reshape(X, [self.batch_size, self.args.board_size, self.args.board_size, self.args.feature_num], name="X")

                self.X = tf.to_float(X)

    def build_model(self):
        board_size = self.args.board_size
        batch_size = self.args.batch_size
        with self.graph.as_default():
            if self.mode == "predict":
                with tf.device('/gpu:%s' % self.gpu_list[0]):
                    logit, _ = self._forward(self.X, batch_size, False, dev='tower0', reuse=None)
                    prob = tf.nn.softmax(logit)
                    self.prob = tf.identity(prob, "policy_output")
                    #self.value = tf.identity(value, "value_output")
            else:
                #TODO:
                #lr = tf.train.piecewise_constant()
                lr = self.args.lr

                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

                X, nxt_move, label, pos = self.sample_buffer.dequeue_many(batch_size)
                n_gpu = len(self.gpu_list)
                batch_X  = tf.split(X, n_gpu)
                batch_nm = tf.split(nxt_move, n_gpu)
                batch_label = tf.split(label, n_gpu)

                tower_grads = []
                tower_loss  = []
                tower_prob  = []
                tower_ce    = []
                tower_mse   = []
                tower_reg   = []

                for i, cur_gpu in enumerate(self.gpu_list):
                    cur_gpu = int(cur_gpu)
                    print ("Build graph on GPU: ", cur_gpu)
                    with tf.device('/device:GPU:%d' % cur_gpu):
                        with tf.name_scope('tower_%d' % cur_gpu) as scope:
                            gpu_batch_size = batch_size / n_gpu # This should be diveded 
                            regularizer = tf.contrib.layers.l2_regularizer(self.args.l2)
                            if i == 0:
                                #logit, prob, v = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=None, regularizer=regularizer)
                                logit, prob = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=None, regularizer=regularizer)
                            else:
                                #logit, prob, v = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=True, regularizer=regularizer)
                                logit, prob = self._forward(batch_X[i], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=True, regularizer=regularizer)

                            tower_prob.append(prob)
                            #ce, mse, reg, loss, kl = self._loss(logit, v, batch_nm[i], batch_label[i], prob, regularizer=regularizer)
                            ce, mse, reg, loss, kl = self._loss(logit, 0, batch_nm[i], batch_label[i], prob, regularizer=regularizer)

                            grads = optimizer.compute_gradients(loss)
                            if grads is None:
                                print (i)
                            tower_grads.append(grads)
                            tower_loss.append(loss)
                            tower_ce.append(ce)
                            tower_mse.append(mse)
                            tower_reg.append(reg)

                            tf.summary.scalar('/gpu:%d/mse'%i, mse)
                            tf.summary.scalar('/gpu:%d/ce'%i,ce)
                            tf.summary.scalar('/gpu:%d/reg'%i,reg)
                            tf.summary.scalar('/gpu:%d/loss'%i,loss)
                            tf.summary.scalar('/gpu:%d/kl'%i,kl)

                self.prob = tf.reshape(tower_prob, [batch_size, board_size * board_size])
                grads = self._average_gradients(tower_grads)
                apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)
                self.train_step = apply_gradient_op
                self.loss = tf.reduce_mean(tower_loss)
                self.ce   = tf.reduce_mean(tower_ce)
                self.mse  = tf.reduce_mean(tower_mse)
                self.reg  = tf.reduce_mean(reg)

                tf.summary.scalar("lr", lr)
                self.summary_step = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            self.all_var = tf.global_variables()

    def push_sample(self, features, nxt_move, label, pos, get_cur_size=False):
        features = self._extract_feature(features, pos)
        feed = {self.features: features, self.nxt_move: nxt_move, self.label: label, self.pos: pos}
        if get_cur_size:
            _, size = self.sess.run([self.feed_step, self.queue_size], feed_dict=feed)
            return size
        else:
            self.sess.run([self.feed_step], feed_dict=feed)
    # For supervised training
    def sl_train(self, summary_writer, features, labels, print_step=100):
        start_time = time.time()
        feed_dict = {self.features: features, self.nxt_move: labels}
        for _ in range(print_step - 1):
            self.sess.run(self.train_step, feed_dict=feed_dict)
        _, loss, ce, mse, reg, prob, summary = self.sess.run([self.train_step, self.loss, self.ce, self.mse, self.reg, self.prob, self.summary_step], feed_dict=feed_dict)
        acc = accuracy_score(np.argmax(labels, 1), np.argmax(prob, 1))
        if self.overall_acc == -1.0:
            self.overall_acc = acc
        else:
            self.overall_acc = 0.999 * self.overall_acc + 0.001 * acc
        global_step = self.sess.run(self.global_step)
        summary_writer.add_summary(summary, global_step)

        duration = time.time() - start_time
        num_sample_per_step = self.batch_size
        sample_per_sec = num_sample_per_step  * print_step / duration
        sec_per_batch = duration / print_step
        format_str = ("global step %d loss %.3f; ce %.3f; mse %.3f; reg %.3f acc %.3f total_acc %.3f; %.1f samples/sec; %.3f sec/batch")
        print (format_str % (global_step, loss, ce, mse, reg, acc, self.overall_acc, sample_per_sec, sec_per_batch))
        sys.stdout.flush()


    def train(self, summary_writer, print_step=100):
        start_time = time.time()
        for _ in range(print_step - 1):
            self.sess.run(self.train_step)
        _, loss, ce, mse, reg, prob, summary = self.sess.run([self.train_step, self.loss, self.ce, self.mse, self.reg, self.prob, self.summary_step])
        global_step = self.sess.run(self.global_step)
        summary_writer.add_summary(summary, global_step)

        duration = time.time() - start_time
        num_sample_per_step = self.batch_size
        sample_per_sec = num_sample_per_step  * print_step / duration
        sec_per_batch = duration / print_step
        format_str = ("global step %d loss %.3f; ce %.3f; mse %.3f; reg %.3f; %.1f samples/sec; %.3f sec/batch")
        print (format_str % (global_step, loss, ce, mse, reg, sample_per_sec, sec_per_batch))
        sys.stdout.flush()

    def get_step(self):
        return self.sess.run(self.global_step)

    def get_all_var(self):
        return self.all_var

    def predict(self, features):
        feed_dict = {self.X: features}
        prob = self.sess.run([self.prob], feed_dict=feed_dict)
        return prob

    #def predict(self, historys):
    #    ori_len = len(historys)
    #    while len(historys) != self.batch_size:
    #        historys.append("")
    #    feed = {self.history: historys}
    #    prob, value = self.sess.run([self.prob, self.value], feed_dict=dict)
    #    return prob[:ori_len, :], value[:ori_len]






            





