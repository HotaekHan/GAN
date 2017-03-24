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
def generator(z):
    net = slim.fully_connected(z, 16384, activation_fn=None)
    net = tf.reshape(net, [-1, 4, 4, 1024])
    net = slim.conv2d_transpose(net, 512, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d_transpose(net, 256, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d_transpose(net, 128, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    g_image = slim.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.tanh)

    return g_image


def discriminator(sample):
    net = slim.conv2d(sample, 128, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 256, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 512, [5, 5], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 1024, [4, 4], stride=2, padding='SAME', activation_fn=slim.nn.relu)
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 1, [4, 4], stride=1, padding='VALID', activation_fn=slim.nn.relu)
    logit = tf.reshape(net, [-1, 1])
    prob = slim.nn.sigmoid(logit)

    return prob, logit