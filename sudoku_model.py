import sys
import numpy as np
import tensorflow as tf
import argparse
import tfutils

import time
from model import model

class sudoku_model(model):
    def __init__(self, args, name, mode, batch_size, gpu_list='0'):
        super().__init__()

    def _forward(self, inputs, batch_size, is_train, device, reuse, regularizer=None, rot_var=None):
        X, _ = inputs
        n_filter=16
        n_extra = 8

        net = tf.contrib.layers.conv2d(X, n_filter, kernel_size=3, biases_initializer=None, activation_fn=None, scope="input", reuse=reuse, weights_regularizer=regularizer)
        new = tfutils.
