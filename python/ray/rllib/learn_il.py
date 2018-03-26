from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random

import pickle
import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.cartpole import CartpoleEncoder
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
    act = action_dist.sample()
    print("IL loss", il_loss)

    # Set up autoencoder loss
    orig_obs = tf.placeholder(tf.float32, [None, 4])
    autoencoder_in = network.last_layer
    if not autoencoder_loss_enabled:
        autoencoder_in = tf.stop_gradient(autoencoder_in)
    recons_obs = slim.fully_connected(
        autoencoder_in, 4,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="fc_autoencoder_out")
    autoencoder_loss = tf.reduce_mean(tf.square(orig_obs - recons_obs))
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
    preprocessor = CartpoleEncoder(env.observation_space, {
        "custom_options": {
            "seed": 0,
            "out_size": 200,
        },
    })
#        "custom_options": {
#            "seed": 0,
#            "noise_size": N,
#            "matrix_size": N,
#            "invert": False,
#        },
#    })

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    vars = TensorFlowVariables(summed_loss, sess)

    data = [json.loads(x) for x in open(PATH).readlines()]
    print("preprocessing data")
    for t in data:
        t["encoded_obs"] = preprocessor.transform(t["obs"])
        t["encoded_next_obs"] = preprocessor.transform(t["new_obs"])
    random.seed(0)
    random.shuffle(data)
    split_point = int(len(data) * 0.9)
    test_batch = data[split_point:]
    data = data[:split_point]
    means = [np.mean([abs(d["obs"][j]) for d in test_batch]) for j in range(4)]
    print("train batch size", len(data))
    print("test batch size", len(test_batch))

    print("start training")
    for i in range(1000):
        il_losses = []
        auto_losses = []
        inv_dyn_losses = []
        errors = [[], [], [], []]
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
            for i in range(len(batch)):
                obs = batch[i]["obs"]
                pred_obs = x[i]
                for j in range(4):
                    errors[j].append(abs(obs[j] - pred_obs[j]))
            il_losses.append(cur_il_loss)
            auto_losses.append(cur_auto_loss)
            inv_dyn_losses.append(cur_inv_dyn_loss)
        for j in range(4):
            errors[j] = np.mean(errors[j])

        test_inv_dyn_loss, test_il_loss, test_auto_loss = sess.run(
            [inv_dyn_loss, il_loss, autoencoder_loss],
            feed_dict={
                observations: [t["encoded_obs"] for t in test_batch],
                expert_actions: [t["action"] for t in test_batch],
                orig_obs: [t["obs"] for t in test_batch],
                next_obs: [t["encoded_next_obs"] for t in test_batch],
            })
        acc = np.mean([np.exp(-l) for l in il_losses])
        auto_loss = np.mean(auto_losses)
        ivd_acc = np.mean([np.exp(-l) for l in inv_dyn_losses])
        ivd_loss = np.mean(inv_dyn_losses)

        # Evaluate IL performance
        rewards = []
        for _ in range(100):
            obs = env.reset()
            reward = 0
            done = False
            while not done:
                action = sess.run(act, feed_dict={observations: [preprocessor.transform(obs)]})[0]
                obs, rew, done, _ = env.step(action)
                reward += rew
            rewards.append(reward)

        reporter(
            timesteps_total=i, mean_loss=np.mean(il_losses) + auto_loss + ivd_loss, info={
                "decoder_reconstruction_error": {
                    "cart_pos": errors[0] / means[0],
                    "pole_angle": errors[1] / means[1],
                    "cart_velocity": errors[2] / means[2],
                    "angle_velocity": errors[3] / means[3],
                },
                "train_il_acc": acc,
                "train_il_loss": np.mean(il_losses),
                "train_auto_loss": auto_loss,
                "train_inv_dyn_loss": ivd_loss,
                "train_inv_dyn_acc": ivd_acc,
                "test_il_mean_reward": np.mean(rewards),
                "test_il_acc": np.exp(-test_il_loss),
                "test_il_loss": test_il_loss,
                "test_auto_loss": test_auto_loss,
                "test_inv_dyn_loss": test_inv_dyn_loss,
                "test_inv_dyn_acc": np.exp(-test_inv_dyn_loss),
            })

        if i % 1 == 0:
            fname = "weights_{}".format(i)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


ray.init()
register_trainable("il", train)
run_experiments({
    "iltrain": {
        "run": "il",
        "config": {
            "N": 200 - 4,
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
