import numpy as np
import tensorflow as tf
import argparse

import os
import time

#Base class for network model
class model_base(object):
    def __init__(self, args, gpu_list):
        self.model_path = args.model_path
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        if not gpu_list is None:
            config.gpu_options.visible_device_list = str(gpu_list)

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        self.cur_checkpoint="None"

    def _loss(self, X, y):
        pass

    def _forward(self, batch_X, is_train=False, reuse=True):
        pass

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(values=grads, axis=0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_vars = (grad, v)
            average_grads.append(grad_and_vars)
        return average_grads

    def build_input(self):
        pass

    def build_model(self):
        pass

    def save_model(self, path=None):
        if path is None:
            path = self.model_path
        self.saver.save(self.sess, os.path.join(path, 'model.ckpt'), global_step=self.global_step)

    # Raise ValueError if incorrect path provided
    def load_model(self, path=None):
        if path is None:
            path = self.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            if ckpt.model_checkpoint_path == self.cur_checkpoint:
                print ("Do not load the same model %s again" % (ckpt.model_checkpoint_path))
            else:
                print ("Load model from %s" % (ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.cur_checkpoint = str(ckpt.model_checkpoint_path)
            return True
        else:
            pass
            #raise ValueError("No Model to Load")

        
            
                    
