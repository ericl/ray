"""Example of how to interface a model with Ray SGD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import ray
from ray.tune import run_experiments
from ray.tune.examples.tune_mnist_ray import *
from ray.experimental.sgd.model import Model
from ray.experimental.sgd.sgd import DistributedSGD

parser = argparse.ArgumentParser()
parser.add_argument("--redis-address", default=None, type=str)
parser.add_argument("--num-iters", default=10000, type=int)
parser.add_argument("--batch-size", default=50, type=int)
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--devices-per-worker", default=1, type=int)
parser.add_argument(
    "--strategy", default="ps", type=str, help="One of 'simple' or 'ps'")
parser.add_argument(
    "--gpu", action="store_true", help="Use GPUs for optimization")


class MNISTModel(Model):
    def __init__(self):
        # Import data
        for _ in range(10):
            try:
                self.mnist = input_data.read_data_sets(
                    "/tmp/tensorflow/mnist/input_data", one_hot=True)
                break
            except Exception as e:
                time.sleep(5)
        if not hasattr(self, "mnist"):
            raise ValueError("Failed to import data", e)

        self.x = tf.placeholder(tf.float32, [None, 784], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
        y_conv, self.keep_prob = deepnn(self.x)

        # Need to define loss and optimizer attributes
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_, logits=y_conv))
        self.optimizer = tf.train.AdamOptimizer(1e-4)

        # For evaluating test accuracy
        correct_prediction = tf.equal(
            tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_feed_dict(self):
        batch = self.mnist.train.next_batch(50)
        return {
            self.x: batch[0],
            self.y_: batch[1],
            self.keep_prob: 0.5,
        }

    def test_accuracy(self):
        return self.accuracy.eval(
            feed_dict={
                self.x: self.mnist.test.images,
                self.y_: self.mnist.test.labels,
                self.keep_prob: 1.0,
            })


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)

    sgd = DistributedSGD(
        lambda w_i, d_i: MNISTModel(),
        num_workers=args.num_workers,
        devices_per_worker=args.devices_per_worker,
        gpu=args.gpu,
        strategy=args.strategy)

    for i in range(args.num_iters):
        if i % 10 == 0:
            start = time.time()
            loss = sgd.step(fetch_stats=True)
            acc = sgd.foreach_model(lambda model: model.test_accuracy())
            print("Iter", i, "loss", loss, "accuracy", acc)
            acc = sgd.foreach_model(lambda model: model.test_accuracy())
            print("acc2", acc)
            print("Time per iteration", time.time() - start)
        else:
            sgd.step()
