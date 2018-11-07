import sys
import numpy as np
import tensorflow as tf
import argparse
import tf_utils

import time
from model import model

class sudoku_model(model):
    def __init__(self, args, mode, gpu_list='0'):
        super().__init__(args, mode, gpu_list)
        self.args = args

    def _forward(self, inputs, batch_size, is_train, dev, reuse, regularizer=None, rot_var=None):
        X = inputs
        n_filter=16
        n_extra = 8

        net = tf.contrib.layers.conv2d(X, n_filter, kernel_size=3, biases_initializer=None, activation_fn=None, scope="input", reuse=reuse, weights_regularizer=regularizer)
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet0")
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet1")
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet2")

        net = tf_utils.conv_bn_relu(net, n_filter, k_size=3, scope="reduce", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regularizer)
        
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet3")
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet4")
        net = tf_utils.resnet_bn_block_normal(net, n_filter, is_train, reuse=reuse, weights_regularizer=regularizer, dev=dev, scope="resnet5")

        # Value
        value = tf_utils.conv_bn_relu(net, 1, 1, scope="value0", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regularizer)
        value = tf.contrib.layers.flatten(value)
        value = tf.contrib.layers.fully_connected(value, 256, scope="value1", reuse=reuse, weights_regularizer=regularizer)
        value = tf.contrib.layers.fully_connected(value, 1, activation_fn=tf.nn.relu, scope="value2", reuse=reuse, weights_regularizer=regularizer)
        value = tf.squeeze(value, axis=1)

        # Policy
        policy = tf_utils.conv_bn_relu(net, 2, 1, scope="policy0", dev=dev, is_train=is_train, reuse=reuse, weights_regularizer=regularizer)
        policy = tf.contrib.layers.flatten(policy)
        logit = tf.contrib.layers.fully_connected(policy, self.args.board_size, scope="policy1", activation_fn=None, reuse=reuse, weights_regularizer=regularizer)
        pred = tf.nn.softmax(logit)

        value = tf.identity(value, "value_output")
        pred = tf.identity(pred, "policy_output")

        return logit, pred, value
