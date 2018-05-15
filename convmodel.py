# -*- coding: utf-8 -*-

# Sample code to use string producer.

#NO APRENDE BIEN, TAMPOCO PUEDO LEER EL MODELO EN EL PLAYER

import tensorflow as tf
import numpy as np


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


num_classes = 3
batch_size = 4

def uanjot(i):
    a=[0,0]
    a.insert(i, 1.)
    return a



# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), uanjot(i)  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 256. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        o5 = tf.layers.conv2d(inputs=o4, filters=64, kernel_size=3, activation=tf.nn.relu)
        o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)
        h = tf.layers.dense(inputs=tf.reshape(o6, [batch_size * 3, 8 * 15 * 64]), units=1024, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["./8data/train/0/*.jpg", "./8data/train/1/*.jpg",
                                                     "./8data/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["./8data/valid/0/*.jpg", "./8data/valid/1/*.jpg",
                                                     "./8data/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["./8data/test/0/*.jpg", "./8data/test/1/*.jpg",
                                                   "./8data/test/2/*.jpg",], batch_size=batch_size)


example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    err = 1000
    err2 = 100

    for _ in range(2000):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            err = err2
            err2 = sess.run(cost_valid)
            print("Error:", err2)



    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    print("----------------------")
    print("     Start test...  ")
    print("----------------------")

    res_test = sess.run(example_batch_test_predicted)
    y_t = sess.run(label_batch_test)
    total = 0

    for lis_res, real_values in zip(res_test, y_t):
        if np.argmax(lis_res) == np.argmax(real_values):
            total += 1

    precision = total / float(len(y_t))
    print ("This net has a precision of", precision)

    coord.request_stop()
    coord.join(threads)