# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'fuhuamosi'


def load_data():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


if __name__ == '__main__':
    print(load_data().train.next_batch(50))
