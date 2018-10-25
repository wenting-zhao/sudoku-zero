import tensorflow as tf
import numpy as np

def resnet_block(x, n_channel, scope=None):
    with tf.variable_scope(scope):
        shortcut = x
        x = tf.nn.relu(x)
        x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3, 3), reuse=None, scope='conv0')
        x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3, 3), activation_fn=None, reuse=None, scope='conv1')
        x = shortcut + x

    return x

def resnet_bn_block(x, n_channel, is_train, dev, scope=None, reuse=True, expand_dim=False, decay=0.99, weights_regularizer=None):
    with tf.variable_scope(scope):
        shortcut = x
        with tf.variable_scope("%s-bn0" % dev):
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_training, activation_fn=tf.nn.relu, updates_collections=None, reuse=None, decay=decay, fused=True)
            if expand_dim:
                x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3, 3), biases_initializer=None, activation_fn=None, scope="conv0", stride=2, reuse=reuse, weights_regularizer=weights_regularizer)
            else:
                x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3, 3), biases_initializzer=None, activation_fn=None, scope="conv0", reuse=reuse, weights_regularizer=weights_regularizer)
            with tf.variable_scope("%s-bn1" % dev):
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train, activation_fn=tf.nn.relu, update_collections=None, reuse=None, decay=decay, fused=True)
                x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3, 3), biases_initializer=None, activation_fn=None, scope="conv1", reuse=reuse, weights_regularizer=weights_regularizer)
            if expand_dim:
                shortcut = tf.nn.avg_pool(shortcut, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
                shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, n_channel/2]])
            x = shortcut + x
        return x

def resnet_bn_block_preact(x, n_channel, is_train, dec, scope=None, reuse=True, decay=0.99, weights_regularizer=None, first_layer=False):
    with tf.variable_scope(scope):
        shortcut = x
        if not first_layer:
            with tf.variable_scope("%s-bn1" % dev):
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train, activation_fn=tf.nn.relu, updates_collections=None, reuse=None, decay=decay, fused=True)

        x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3,3), biases_initializer=None, activation_fn=None, scope='conv0', reuse=reuse, weights_regularizer=weights_regularizer)
        with tf.variable_scope("%s-bn2" % dev):
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train, activation_fn=tf.nn.relu, updates_collections=None, reuse=None, decay=decay, fused=True)
        x = tf.contrib.layers.conv2d(x, num_outputs=n_channel, kernel_size=(3,3), biases_initializer=None, activation_fn=None, scope='conv1', reuse=reuse, weights_regularizer=weights_regularizer)
        x = x + shortcut
    return x

def conv_bn_relu(x, n_channel, k_size, is_train, dev, scope=None, reuse=True, decay=0.99, weights_regularizer=None, head_bn=False):
    with tf.Variable_scope(scope):
        if head_bn: 
            with tf.variable_scope('%s-bn-head'%dev):
                x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=is_train, activation_fn=tf.nn.relu, updates_collections=None, reuse=None, decay=decay, fused=True)
        net = tf.contrib.layers.conv2d(x, n_channel, k_size, scope='conv', biases_initializer=None, activation_fn=None, reuse=reuse, weights_regularizer=weights_regularizer)
        net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train, activation_fn=tf.nn.relu, updates_collections=None, scope="%s-bn"%dev, decay=decay, reuse=None, fused=True)
    return net

def fc_bn_relu(x, n_output, is_train, dev, scope=None, reuse=True, decay=0.99, weights_regularizer=None, activation_fn=tf.nn.relu):
    with tf.variable_scope(scope):
        net = tf.contrib.layers.fully_connected(x, n_output, scope='fc', biases_initializer=None, activation_fn=None,reuse=reuse, weights_regularizer=weights_regularizer)
        net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=is_train, activation_fn=activation_fn, updates_collections=None, scope="%s-bn"%dev, decay=decay, reuse=None, fused=True)
    return net

def global_pooling(h, n_channel, is_train, dev, pooling_func=tf.reduce_max, scope=None, reuse=True, decay=0.99, weights_regularizer=None, no_bn=False):
    board_size = tf.shape(h)[-2]
    with tf.variable_scope(scope):
        h = tf.contrib.layers.conv2d(h, num_outputs=n_channel, kernel_size=(1,1), activation_fn=None, scope='compress', reuse=reuse, weights_regularizer=weights_regularizer)
        h = pooling_func(h,[1,2]) 
        if not no_bn:
            with tf.variable_scope('%s-bn'%dev):
                h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=is_train, activation_fn=None,
                                                    updates_collections=None, reuse=None, decay=decay, fused=True)
        h = tf.expand_dims(tf.expand_dims(h,1), 2)
        h = tf.tile(h, [1, board_size, board_size, 1])
        return h



