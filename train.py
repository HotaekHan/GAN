import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data_reader as reader
import networks as net

def sample_Z(batch_size, dimension):
    # return np.random.uniform(-1., 1., size=[batch_size, dimension])
    return np.random.normal(loc=0.0, scale=1.0, size=[batch_size, dimension])

def plot(samples):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64), cmap='gray')

    return fig


def main():

    log_dir = "log/model_without_noise.ckpt"
    # train_dir_path = "D:\\Projects\\RFR\\Dataset\\HUD_1st\\converted\\Training"
    train_dir_path = "C:\\Users\\VTouch\\Documents\\Experiment\\TensorFlow\\Train_Face"
    # train_dir_path = "C:\\Users\\VTouch\\Documents\\Data\\NYU_Hand_Dataset\\dataset\\cropped_train\\syth"
    # train_dir_path = "C:\\Users\\VTouch\\Documents\\Data\\CelebA_Dataset\\img_align_celeba"

    # train_dir_path = "C:\\Users\\VTouch\\Documents\\Data\\thumbnails_features_deduped_publish\\thumbnails_features_deduped_publish"

    resized_width = 64
    resized_height = 64

    size_of_batch = 64
    epoch = 100000
    Z_dim = 100

    images_list = reader.read_data_list(train_dir_path, image_ext=".png")

    image_name_queue = tf.train.string_input_producer(images_list)

    image_reader = tf.WholeFileReader()

    key, image_value = image_reader.read(image_name_queue)

    image_decoded = tf.image.decode_png(image_value, channels=1)

    image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.BILINEAR)

    # normalize 0 to 1
    # image = tf.scalar_mul(1.0 / 255.0, image)

    # normalize -1 to 1
    image = tf.scalar_mul(2.0 / 255.0, image) - tf.constant(1.0)

    x = tf.train.shuffle_batch(tensors=[image], batch_size=size_of_batch, num_threads=4, capacity=30000, min_after_dequeue=64)

    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    g_image = net.generator(Z, is_train=True)

    d_fake, d_logit_fake = net.discriminator(g_image, is_train=True)
    d_real, d_logit_real = net.discriminator(x, is_train=True, is_reuse=True)


    # d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
    # g_loss = -tf.reduce_mean(tf.log(d_fake))

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
    d_loss = D_loss_real + D_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

    trainable_vars = tf.trainable_variables()

    d_vars = [var for var in trainable_vars if "discriminator" in var.name]
    g_vars = [var for var in trainable_vars if "generator" in var.name]

    d_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=g_vars)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    i = 0

    # for var_tf in tf.global_variables():
    #     print(var_tf.name)

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for iter_epoch in range(epoch):
            if iter_epoch % 100 == 0:
                samples = sess.run(g_image, feed_dict={Z: sample_Z(size_of_batch, Z_dim)})

                samples += 1.0
                samples = samples * (255.0 / 2.0)
                samples = samples.astype(np.uint8)

                # samples = samples * 255.0

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