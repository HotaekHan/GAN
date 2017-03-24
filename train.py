import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data_reader as reader
import networks as net


def sample_Z(batch_size, dimension):
    return np.random.uniform(-1., 1., size=[batch_size, dimension])

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64), cmap='Greys_r')

    return fig


def main():

    log_dir = "log/model_without_noise.ckpt"
    train_dir_path = "D:\\Projects\\RFR\\Dataset\\HUD_1st\\converted\\Training"

    resized_width = 64
    resized_height = 64

    size_of_batch = 128
    epoch = 10000
    Z_dim = 100

    images_list = reader.read_data_list(train_dir_path, image_ext=".png")

    image_name_queue = tf.train.string_input_producer(images_list)

    image_reader = tf.WholeFileReader()

    _, image_value = image_reader.read(image_name_queue)

    image_decoded = tf.image.decode_png(image_value, channels=1)

    image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.BILINEAR)

    # normalize 0 to 1
    # image = tf.scalar_mul(1.0 / 255.0, image)

    # normalize -1 to 1
    image = tf.scalar_mul(2.0 / 255.0, image) - tf.constant(1.0)

    x = tf.train.shuffle_batch(tensors=[image], batch_size=size_of_batch, num_threads=4, capacity=50000, min_after_dequeue=10000)

    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    g_image = net.generator(Z)

    d_real, d_logit_real = net.discriminator(x)
    d_fake, d_logit_fake = net.discriminator(g_image)

    d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
    g_loss = -tf.reduce_mean(tf.log(d_fake))

    # d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(d_loss)
    # g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001).minimize(g_loss)

    d_optimizer = tf.train.AdamOptimizer().minimize(d_loss)
    g_optimizer = tf.train.AdamOptimizer().minimize(g_loss)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    i = 0

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for iter_epoch in range(epoch):
            if iter_epoch % 1000 == 0:
                samples = sess.run(g_image, feed_dict={Z: sample_Z(16, Z_dim)})

                samples += 1.0
                samples = samples * (255.0 / 2.0)

                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

            _, batch_d_loss = sess.run([d_optimizer, d_loss], feed_dict={Z: sample_Z(size_of_batch, Z_dim)})
            _, batch_g_loss = sess.run([g_optimizer, g_loss], feed_dict={Z: sample_Z(size_of_batch, Z_dim)})



            if iter_epoch % 100 == 0:
                print('Iter: {}'.format(iter_epoch))
                print('D loss: {:.4}'.format(batch_d_loss))
                print('G_loss: {:.4}'.format(batch_g_loss))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()




# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)
#
#
# X = tf.placeholder(tf.float32, shape=[None, 784])
#
# D_W1 = tf.Variable(xavier_init([784, 128]))
# D_b1 = tf.Variable(tf.zeros(shape=[128]))
#
# D_W2 = tf.Variable(xavier_init([128, 1]))
# D_b2 = tf.Variable(tf.zeros(shape=[1]))
#
# theta_D = [D_W1, D_W2, D_b1, D_b2]
#
#
# Z = tf.placeholder(tf.float32, shape=[None, 100])
#
# G_W1 = tf.Variable(xavier_init([100, 128]))
# G_b1 = tf.Variable(tf.zeros(shape=[128]))
#
# G_W2 = tf.Variable(xavier_init([128, 784]))
# G_b2 = tf.Variable(tf.zeros(shape=[784]))
#
# theta_G = [G_W1, G_W2, G_b1, G_b2]
#
#
#
#
#
# def generator(z):
#     G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#     G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#     G_prob = tf.nn.sigmoid(G_log_prob)
#
#     return G_prob
#
#
# def discriminator(x):
#     D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
#     D_logit = tf.matmul(D_h1, D_W2) + D_b2
#     D_prob = tf.nn.sigmoid(D_logit)
#
#     return D_prob, D_logit
#
#
# def plot(samples):
#     fig = plt.figure(figsize=(4, 4))
#     gs = gridspec.GridSpec(4, 4)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#     return fig
#
#
# G_sample = generator(Z)
# D_real, D_logit_real = discriminator(X)
# D_fake, D_logit_fake = discriminator(G_sample)
#
# # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# # G_loss = -tf.reduce_mean(tf.log(D_fake))
#
# # Alternative losses:
# # -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_real, tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_logit_fake, tf.ones_like(D_logit_fake)))
#
# D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
#
# mb_size = 128
# Z_dim = 100
#
# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# if not os.path.exists('out/'):
#     os.makedirs('out/')
#
# i = 0
#
# for it in range(1000000):
#     if it % 1000 == 0:
#         samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
#
#         fig = plot(samples)
#         plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#         i += 1
#         plt.close(fig)
#
#     X_mb, _ = mnist.train.next_batch(mb_size)
#
#     _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
#     _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
#
#     if it % 1000 == 0:
#         print('Iter: {}'.format(it))
#         print('D loss: {:.4}'. format(D_loss_curr))
#         print('G_loss: {:.4}'.format(G_loss_curr))
# print()