# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from app.experts.data_preprocess import load_data

__author__ = 'fuhuamosi'


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def evaluate_lenet5(learning_rate=0.1, n_epochs=200, nkerns=None, batch_size=500):
    mnist = load_data()
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    w_conv1 = weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = bias_variable(shape=[32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    w_conv2 = weight_variable(shape=[5, 5, 32, 64])
    b_conv2 = bias_variable(shape=[64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    w_fc1 = weight_variable(shape=[7 * 7 * 64, 1024])
    b_fc1 = bias_variable(shape=[1024])
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable(shape=[1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(y_conv),
                                                  reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], label: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], label: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, label: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    evaluate_lenet5()
    # sess = tf.InteractiveSession()
    # w = weight_variable(shape=[2, 2])
    # sess.run(tf.initialize_all_variables())
    # print(w.eval())
