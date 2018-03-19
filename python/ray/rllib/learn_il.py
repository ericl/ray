from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import pickle
import numpy as np
import gym
import tensorflow as tf

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.cartpole import MakeCartpoleHarder
from ray.tune import run_experiments, register_trainable, grid_search

PATH = os.path.expanduser("~/Desktop/cartpole-expert.json")


def train(config, reporter):
    N = config.get("N", 500)
    BATCH_SIZE = config.get("batch_size", 128)

    observations = tf.placeholder(tf.float32, [None, N + 4])
    expert_actions = tf.placeholder(tf.int32, [None])
    network = FullyConnectedNetwork(observations, 2, config.get("model", {}))
    action_dist = Categorical(network.outputs)

    loss = -tf.reduce_mean(action_dist.logp(expert_actions))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    env = gym.make("CartPole-v0")
    preprocessor = MakeCartpoleHarder(env.observation_space, {
        "custom_options": {
            "seed": 0,
            "noise_size": N,
            "matrix_size": N,
            "invert": False,
        },
    })

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    vars = TensorFlowVariables(loss, sess)

    data = [json.loads(x) for x in open(PATH).readlines()]
    print("preprocessing data")
    for t in data:
        t["obs"] = preprocessor.transform(t["obs"])

    print("start training")
    for i in range(1000):
        losses = []
        for _ in range(len(data) // BATCH_SIZE):
            batch = np.random.choice(data, BATCH_SIZE)
            cur_loss, _ = sess.run([loss, train_op], feed_dict={
                observations: [t["obs"] for t in batch],
                expert_actions: [t["action"] for t in batch],
            })
            losses.append(cur_loss)
        acc = np.mean([np.exp(-l) for l in losses])
        reporter(timesteps_total=i, mean_accuracy=acc)
        if i % 1 == 0:
            fname = "weights_{}".format(i)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


ray.init()
register_trainable("il", train)
run_experiments({
    "il": {
        "run": "il",
        "config": {
            "N": 500,
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 8],
            },
        },
    }
})
