from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import pickle
import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.cartpole import MakeCartpoleHarder
from ray.rllib.models.misc import normc_initializer
from ray.tune import run_experiments, register_trainable, grid_search

PATH = os.path.expanduser("~/Desktop/cartpole-expert.json")


def train(config, reporter):
    N = config.get("N", 500)
    BATCH_SIZE = config.get("batch_size", 128)
    il_loss_enabled = config.get("il_loss", True)
    autoencoder_loss_enabled = config.get("autoencoder_loss", False)
    inv_dyn_loss_enabled = config.get("inv_dynamics_loss", False)

    # Set up decoder network
    observations = tf.placeholder(tf.float32, [None, N + 4])
    network = FullyConnectedNetwork(observations, 2, config.get("model", {}))

    # Set up IL loss
    expert_actions = tf.placeholder(tf.int32, [None])
    action_dist = Categorical(network.outputs)
    if il_loss_enabled:
        il_loss = -tf.reduce_mean(action_dist.logp(expert_actions))
    else:
        il_loss = tf.constant(0.0)
    print("IL loss", il_loss)

    # Set up autoencoder loss
    orig_obs = tf.placeholder(tf.float32, [None, 4])
    recons_obs = slim.fully_connected(
        network.last_layer, 4,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="fc_autoencoder_out")
    if autoencoder_loss_enabled:
        autoencoder_loss = tf.reduce_mean(tf.square(orig_obs - recons_obs))
    else:
        autoencoder_loss = tf.constant(0.0)
    print("Autoencoder loss", autoencoder_loss)

    # Set up inverse dynamics loss
    tf.get_variable_scope()._reuse = tf.AUTO_REUSE
    next_obs = tf.placeholder(tf.float32, [None, N + 4])
    network2 = FullyConnectedNetwork(next_obs, 2, config.get("model", {}))
    fused = tf.concat([network.last_layer, network2.last_layer], axis=1)
    fused2 = slim.fully_connected(
        fused, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="inv_dyn_pred1")
    predicted_action = slim.fully_connected(
        fused, 2,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="inv_dyn_pred2")
    inv_dyn_action_dist = Categorical(predicted_action)
    if inv_dyn_loss_enabled:
        inv_dyn_loss = -tf.reduce_mean(
            inv_dyn_action_dist.logp(expert_actions))
    else:
        inv_dyn_loss = tf.constant(0.0)
    print("Inv Dynamics loss", inv_dyn_loss)

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer()
    summed_loss = autoencoder_loss + il_loss + inv_dyn_loss
    train_op = optimizer.minimize(summed_loss)

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

    vars = TensorFlowVariables(summed_loss, sess)

    data = [json.loads(x) for x in open(PATH).readlines()]
    print("preprocessing data")
    for t in data:
        t["encoded_obs"] = preprocessor.transform(t["obs"])
        t["encoded_next_obs"] = preprocessor.transform(t["new_obs"])

    print("start training")
    for i in range(1000):
        il_losses = []
        auto_losses = []
        inv_dyn_losses = []
        for _ in range(len(data) // BATCH_SIZE):
            batch = np.random.choice(data, BATCH_SIZE)
            x, cur_inv_dyn_loss, cur_il_loss, cur_auto_loss, _ = sess.run(
                [recons_obs, inv_dyn_loss, il_loss, autoencoder_loss, train_op],
                feed_dict={
                    observations: [t["encoded_obs"] for t in batch],
                    expert_actions: [t["action"] for t in batch],
                    orig_obs: [t["obs"] for t in batch],
                    next_obs: [t["encoded_next_obs"] for t in batch],
                })
#            print(x[0], batch[0]["obs"])
            il_losses.append(cur_il_loss)
            auto_losses.append(cur_auto_loss)
            inv_dyn_losses.append(cur_inv_dyn_loss)
        acc = np.mean([np.exp(-l) for l in il_losses])
        auto_loss = np.mean(auto_losses)
        ivd_acc = np.mean([np.exp(-l) for l in inv_dyn_losses])
        ivd_loss = np.mean(inv_dyn_losses)
        reporter(
            timesteps_total=i, mean_accuracy=acc, mean_loss=auto_loss + ivd_loss, info={
                "il_loss": np.mean(il_losses),
                "auto_loss": auto_loss,
                "inv_dyn_loss": ivd_loss,
                "inv_dyn_acc": ivd_acc,
            })
        if i % 1 == 0:
            fname = "weights_{}".format(i)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


ray.init()
register_trainable("il", train)
run_experiments({
    "autoencoder": {
        "run": "il",
        "config": {
            "N": 500,
            "model": {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [256, 8],
            },
            "il_loss": False,
            "autoencoder_loss": False,
            "inv_dynamics_loss": True,
        },
    }
})
