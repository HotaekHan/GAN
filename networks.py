import tensorflow.contrib.slim as slim
import tensorflow as tf


# vanilla GAN
# def generator(z):
#     net = slim.fully_connected(z, 512, activation_fn=slim.nn.tanh)
#     net = slim.batch_norm(net)
#     net = slim.fully_connected(net, 1024, activation_fn=slim.nn.tanh)
#     net = slim.batch_norm(net)
#     g_image = slim.fully_connected(net, 1600, activation_fn=slim.nn.tanh)
#
#     return g_image
#
#
# def discriminator(sample):
#     net = slim.fully_connected(sample, 512, activation_fn=slim.nn.tanh)
#     net = slim.batch_norm(net)
#     prediction = slim.fully_connected(net, 1, activation_fn=slim.nn.sigmoid)
#
#     return prediction


# DCGAN
def generator(z, is_train, is_reuse=None):

    with tf.variable_scope("generator", reuse=is_reuse):

        net = slim.fully_connected(z, 16384, activation_fn=None, reuse=is_reuse, scope='g_h0_lin'
                                   , weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = tf.reshape(net, [-1, 4, 4, 1024], name='g_h0_reshape')
        net = slim.batch_norm(net, activation_fn=slim.nn.relu, is_training=is_train,
                              updates_collections=None, reuse=is_reuse, scope='g_h0_batch_norm')

        net = slim.conv2d_transpose(net, 512, [5, 5], stride=2, padding='SAME', activation_fn=None,
                                    reuse=is_reuse, scope='g_h1_deconv2d',
                                    weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=slim.nn.relu, is_training=is_train,
                              updates_collections=None, reuse=is_reuse, scope='g_h1_batch_norm')

        net = slim.conv2d_transpose(net, 256, [5, 5], stride=2, padding='SAME', activation_fn=None,
                                    reuse=is_reuse, scope='g_h2_deconv2d'
                                    ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=slim.nn.relu, is_training=is_train,
                              updates_collections=None, reuse=is_reuse, scope='g_h2_batch_norm')

        net = slim.conv2d_transpose(net, 128, [5, 5], stride=2, padding='SAME', activation_fn=None,
                                    reuse=is_reuse, scope='g_h3_deconv2d'
                                    ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=slim.nn.relu, is_training=is_train,
                              updates_collections=None, reuse=is_reuse, scope='g_h3_batch_norm')

        logits = slim.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=None,
                                       reuse=is_reuse, scope='g_h4_deconv2d'
                                       ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        g_image = tf.nn.tanh(logits)

    return g_image


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, x * leak)


def discriminator(sample, is_train, is_reuse=None):

    with tf.variable_scope("discriminator", reuse=is_reuse, values=[sample]):

        net = slim.conv2d(sample, 64, [5, 5], stride=2, padding='SAME', activation_fn=None,
                          reuse=is_reuse, scope='d_h0_conv2d'
                          ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = leaky_relu(net)

        # net = slim.batch_norm(net)
        net = slim.conv2d(net, 128, [5, 5], stride=2, padding='SAME', activation_fn=None,
                          reuse=is_reuse, scope='d_h1_conv2d'
                          ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=None, is_training=is_train, updates_collections=None,
                              reuse=is_reuse, scope='d_h1_batch_norm')
        net = leaky_relu(net)

        net = slim.conv2d(net, 256, [5, 5], stride=2, padding='SAME', activation_fn=None,
                          reuse=is_reuse, scope='d_h2_conv2d'
                          ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=None, is_training=is_train, updates_collections=None,
                              reuse=is_reuse, scope='d_h2_batch_norm')
        net = leaky_relu(net)

        net = slim.conv2d(net, 512, [4, 4], stride=2, padding='SAME', activation_fn=None,
                          reuse=is_reuse, scope='d_h3_conv2d'
                          ,weights_initializer=tf.random_normal_initializer(stddev=0.02))
        net = slim.batch_norm(net, activation_fn=None, is_training=is_train, updates_collections=None,
                              reuse=is_reuse, scope='d_h3_batch_norm')
        net = leaky_relu(net)

        # net = slim.conv2d(net, 1, [4, 4], stride=1, padding='VALID', activation_fn=slim.nn.relu)
        net = slim.flatten(net, scope='d_h4_flatten')
        logit = slim.fully_connected(net, 1, activation_fn=None, reuse=is_reuse, scope='d_h4_lin'
                                     ,weights_initializer=tf.random_normal_initializer(stddev=0.02))

        # logit = tf.reshape(net, [-1, 1])
        prob = slim.nn.sigmoid(logit)

    return prob, logit