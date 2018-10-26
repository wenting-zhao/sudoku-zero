import numpy as np
import tensorflow as tf
import argparse
import time

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

    def _loss(self, logit, value, nxt_move, label, prob, regularizer):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=nxt_move)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        v_diff = tf.squared_difference(value, label)
        v_loss = tf.reduce_mean(v_diff)

        reg_variables_gradients = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.add_n(reg_variables_gradients)

        eps = 1e-6
        kl_divergence = tf.reduce_sum(nxt_move * tf.log((nxt_move + eps) / (prob + eps)), axis=1)
        kl_mean = tf.reduce_mean(kl_divergence)

        return cross_entropy_mean, v_loss, reg_term, v_loss + cross_entropy_mean + reg_term, kl_mean

    #def _forward(self, inputs, batch_size, is_train, device, reuse, regularizer=None, rot_var=None):
    #        X, _ = inputs
    #        n_filter=16
    #        n_extra = 8

    #        net = tf.contrib.layers.conv2d(X, n_filter, kernel_size=3, biases_initializer=None, activation_fn=None, scope="input", reuse=reuse, weights_regularizer=regularizer)
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet0")
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet1")
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet2")

    #        net = tfutils.conv_bn_relu(net, n_filter, kernel_size=3, scope="reduce", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regularizer)
    #        
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet3")
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet4")
    #        net = tfutils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet5")

    #        # Value
    #        value = tfutils.conv_bn_relu(net, 1, 1, scope="value0", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regulrizer)
    #        value = tf.contrib.layers.flatten(value)
    #        value = tf.contrib.layers.fully_connected(value, 256, scope="value1", reuse=reuse, weights_regularizer=regularizer)
    #        value = tf.contrib.layers.fully_connected(value, 1, activation_fn=tf.nn.relu, scope="value2", reuse=reuse, weights_regularizer=regularizer)
    #        value = tf.squeeze(value, axis=1)

    #        # Policy
    #        policy = tfutils.conv_bn_relu(net, 2, 1, scope="policy0", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regularizer)
    #        policy = tf.contrib.layers.flatten(policy)
    #        logit = tf.conrib.layers.fully_connected(policy, self.args.board_size ** 2 + 1, scope="policy1", activation_fn=None, reuse=reuse, weights_regularizer=regularizer)
    #        pred = tf.nn.softmax(logit)

    #        value = tf.idnetity(value, "value_output")
    #        pred = tf.identity(pred, "policy_output")

    #        return logit, pred, value

    def _extract_feature(self, history):
        # JUST FOR TEST:
        return history, history
        # TODO:
        pass

    def preprocess(self):
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='steps', trainable=False)
            if self.mode == "train":
                #history = self.history = _placeholder(tf.string, name="historys")
                nxt_move = self.nxt_move = _placeholder(name="nxt_move")
                label = self.label = _placeholder(name="label")
                # TODO: write extract feature as a tfop 
                #features, search_prob = self._extract_feature(history)
                #features, search_prob = history, history
                features = self.features = _placeholder(shape=[self.args.board_size, self.args.board_size, 10], name="features")
                # TODO: remove search prob
                search_prob = features

                self.sample_buffer = sample_buffer = tf.RandomShuffleQueue(capacity=self.args.buffer_size, min_after_dequeue=self.args.min_buffer_size, 
                                                                            shapes=[[self.args.board_size, self.args.board_size, self.args.feature_num], [self.args.board_size, self.args.board_size, self.args.feature_num], [self.args.board_size ** 2 + 1], []], 
                                                                            dtypes=[tf.float32, tf.float32, tf.float32, tf.float32])

                self.feed_step = sample_buffer.enqueue((features, search_prob, nxt_move, label))
                self.queue_size = sample_buffer.size()
            else:
                if self.mode == "predict":
                    X = []
                    pi = []
                    history = self.history = _placeholder(tf.string, name='history')
                    for i in range(self.batch_size):
                        # TODO: write extract feature as a tfop 
                        #feature, search_prob = _extract_feature(history[i])
                        features, search_prob = history, history
                        X.append(features)
                        pi.append(search_prob)
                    X = tf.reshape(X, [self.batch_size, self.args.board_size, self.args.board_size, self.args.feature_num], name="X")
                    pi = tf.reshape(pi, [self.batch_size, self.args.board_size, self.args.board_size, 1], name="pi")

                self.X = X
                self.pi = tf.reshape(pi, [self.batch_size, self.args.board_size, self.args.board_size])

    def build_model(self):
        board_size = self.args.board_size
        batch_size = self.args.batch_size
        with self.graph.as_default():
            if self.mode == "predict":
                with tf.device('/gpu:%s' % self.gpu_list[0]):
                    logit, _, value = self._forward([self.X, self.pi], batch_size, False, dev='tower0', reuse=None)
                    prob = tf.nn.softmax(logit)
                    self.prob = tf.identity(prob, "policy_output")
                    self.value = tf.identity(value, "value_output")
            else:
                #TODO:
                #lr = tf.train.piecewise_constant()
                lr = self.args.lr

                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

                X, pi, nxt_move, label = self.sample_buffer.dequeue_many(batch_size)
                n_gpu = len(self.gpu_list)
                batch_X  = tf.split(X, n_gpu)
                batch_pi = tf.split(pi, n_gpu)
                batch_nm = tf.split(nxt_move, n_gpu)
                batch_label = tf.split(label, n_gpu)

                tower_grads = []
                tower_loss  = []
                tower_prob  = []

                for i, cur_gpu in enumerate(self.gpu_list):
                    cur_gpu = int(cur_gpu)
                    with tf.device('/gpu:%d' % cur_gpu):
                        with tf.name_scope('tower_%d' % cur_gpu) as scope:
                            gpu_batch_size = batch_size / n_gpu # This should be diveded 
                            regularizer = tf.contrib.layers.l2_regularizer(self.args.l2)
                            if i == 0:
                                logit, prob, v = self._forward([batch_X[i], batch_pi[i]], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=None, regularizer=regularizer)
                            else:
                                logit, prob, v = self._forward([batch_X[i], batch_pi[i]], gpu_batch_size, is_train=True, dev="tower%d"%i, reuse=True, regularizer=regularizer)

                            tower_prob.append(prob)
                            ce, mse, reg, loss, kl = self._loss(logit, v, batch_nm[i], batch_label[i], prob, regularizer=regularizer)

                            grads = optimizer.compute_gradients(loss)
                            tower_grads.append(grads)
                            tower_loss.append(loss)

                            tf.summary.scalar('/gpu:%d/mse'%i, mse)
                            tf.summary.scalar('/gpu:%d/ce'%i,ce)
                            tf.summary.scalar('/gpu:%d/reg'%i,reg)
                            tf.summary.scalar('/gpu:%d/loss'%i,loss)
                            tf.summary.scalar('/gpu:%d/kl'%i,kl)

                self.prob = tf.reshape(tower_prob, [batch_size, board_size**2 + 1])
                grads = self._average_gradients(tower_grads)
                apply_gradient_op = optimizer.apply_gradients(grads, global_step=self.global_step)
                self.train_step = apply_gradient_op

                tf.summary.scalar("lr", lr)
                self.summary_step = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            self.all_var = tf.global_variables()

    def push_sample(self, features, nxt_move, label, get_cur_size=False):
        feed = {self.features: features, self.nxt_move: nxt_move, self.label: label}
        if get_cur_size:
            _, size = self.sess.run([self.feed_step, self.queue_size], feed_dict=feed)
            return size
        else:
            self.sess.run([self.feed_step], feed_dict=feed)

    def train(self, summary_writer, print_step=100):
        start_time = time.time()
        for _ in range(print_step - 1):
            self.sess.run(self.train_step)
        _, loss, summary = self.sess.run([self.train_step, self.loss, self.summary_step])
        global_step = self.sess.run(self.global_step)
        summary_writer.add_summary(summary, global_step)

        duration = time.time() - start_time
        num_sample_per_step = self.batch_size
        sample_per_sec = num_sample_per_step  * print_step / duration
        sec_per_batch = duration / print_step
        format_str = ("global step %d loss %.3f; &.1f samples/sec; %.3f sec/batch")
        print (format_str % (global_step, loss, sample_per_sec, sec_per_batch))
        sys.stdout.flush()

    def get_step(self):
        return self.sess.run(self.global_step)

    def get_all_var(self):
        return self.all_var

    def predict(self, historys):
        ori_len = len(historys)
        while len(historys) != self.batch_size:
            historys.append("")
        feed = {self.history: historys}
        prob, value = self.sess.run([self.prob, self.value], feed_dict=dict)
        return prob[:ori_len, :], value[:ori_len]






            





