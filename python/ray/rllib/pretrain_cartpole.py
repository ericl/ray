from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import deque
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
from ray.rllib.models.visionnet import VisionNetwork
from ray.rllib.cartpole import ImageCartPole, CartpoleEncoder, parser
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.render_cartpole import render_frame
from ray.tune import run_experiments, register_trainable, grid_search


def make_net(inputs, h_size, image, config):
    if image:
        network = VisionNetwork(inputs, h_size, config.get("model", {}))
        feature_layer = network.outputs
        action_layer = slim.fully_connected(
            feature_layer, 2,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None, scope="action_layer_out")
    else:
        network = FullyConnectedNetwork(inputs, 2, config.get("model", {}))
        feature_layer = network.last_layer
        action_layer = network.outputs
    assert feature_layer.shape[1:] == (h_size,), feature_layer
    assert action_layer.shape[1:] == (2,), action_layer
    return feature_layer, action_layer


def train(config, reporter):
    k = 4
    h_size = 8
    data = config["data"]
    mode = config["mode"]
    env_config = config["env_config"]
    image = config.get("image", False)
    out_size = config.get("out_size", 200)
    batch_size = config.get("batch_size", 128)
    il_loss_enabled = mode == "il"
    autoencoder_loss_enabled = mode == "oracle"
    ivd_loss_enabled = mode in ["ivd", "ivd_fwd"]
    forward_loss_enabled = mode in ["fwd", "ivd_fwd"]
    assert il_loss_enabled or autoencoder_loss_enabled or ivd_loss_enabled or forward_loss_enabled

    # Set up decoder network
    if image:
        observations = tf.placeholder(tf.float32, [None, 80, 80, k])
    else:
        observations = tf.placeholder(tf.float32, [None, out_size])
    feature_layer, action_layer = make_net(observations, h_size, image, config)

    # Set up IL loss
    expert_actions = tf.placeholder(tf.int32, [None])
    action_dist = Categorical(action_layer)
    if il_loss_enabled:
        il_loss = -tf.reduce_mean(action_dist.logp(expert_actions))
    else:
        il_loss = tf.constant(0.0)
    act = action_dist.sample()
    print("IL loss", il_loss)

    # Set up autoencoder loss
    orig_obs = tf.placeholder(tf.float32, [None, 4])
    autoencoder_in = feature_layer
    if not autoencoder_loss_enabled:
        autoencoder_in = tf.stop_gradient(autoencoder_in)
    recons_obs = slim.fully_connected(
        autoencoder_in, 4,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="fc_autoencoder_out")
    autoencoder_loss = tf.reduce_mean(tf.squared_difference(orig_obs, recons_obs))
    print("Autoencoder loss", autoencoder_loss)

    # Set up inverse dynamics loss
    tf.get_variable_scope()._reuse = tf.AUTO_REUSE
    if image:
        next_obs = tf.placeholder(tf.float32, [None, 80, 80, k])
    else:
        next_obs = tf.placeholder(tf.float32, [None, out_size])
    feature_layer2, _ = make_net(next_obs, h_size, image, config)
    fused = tf.concat([feature_layer, feature_layer2], axis=1)
    fused2 = slim.fully_connected(
        fused, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="ivd_pred1")
    predicted_action = slim.fully_connected(
        fused, 2,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="ivd_pred_out")
    ivd_action_dist = Categorical(predicted_action)
    if ivd_loss_enabled:
        ivd_loss = -tf.reduce_mean(
            ivd_action_dist.logp(expert_actions))
    else:
        ivd_loss = tf.constant(0.0)
    print("Inv Dynamics loss", ivd_loss)

    # Set up forward loss
    fwd1 = slim.fully_connected(
        feature_layer, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="fwd1")
    fwd2 = slim.fully_connected(
        fwd1, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="fwd2")
    fwd_out = slim.fully_connected(
        fwd2, h_size,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="fwd_out")
    if forward_loss_enabled:
        fwd_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(feature_layer2), fwd_out)) * 0.01
    else:
        fwd_loss = tf.constant(0.0)

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer()
    summed_loss = autoencoder_loss + il_loss + ivd_loss + fwd_loss
    train_op = optimizer.minimize(summed_loss)

    env = gym.make("CartPole-v0")
    if args.image:
        env = ImageCartPole(env, k, env_config)
        preprocessor = NoPreprocessor(env.observation_space, {})
    else:
        preprocessor = CartpoleEncoder(env.observation_space, {
            "custom_options": {
                "seed": 0,
                "out_size": 200,
            },
        })

    tf_config = tf.ConfigProto(**{
        "gpu_options": {
            "allow_growth": True,
            "per_process_gpu_memory_fraction": 0.3,
        },
    })
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("CUDA: " + os.environ.get("CUDA_VISIBLE_DEVICES"))
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    vars = TensorFlowVariables(summed_loss, sess)

    data = [json.loads(x) for x in open(data).readlines()]
    print("preprocessing data")
    if args.image:
        frames = deque([], maxlen=k)
        data_out = []
        for t in data:
            ok = len(frames) >= k
            if len(frames) == 0:
                frames.append(render_frame(t["obs"], config["env_config"]))
            if ok:
                t["encoded_obs"] = np.concatenate(frames, axis=2)
            frames.append(render_frame(t["new_obs"], config["env_config"]))
            if ok:
                t["encoded_next_obs"] = np.concatenate(frames, axis=2)
                data_out.append(t)
            if t["done"]:
                frames.clear()
            if len(data_out) % 1000 == 0:
                print("Loaded frames", len(data_out))
        data = data_out
    else:
        for t in data:
            t["encoded_obs"] = preprocessor.transform(t["obs"])
            t["encoded_next_obs"] = preprocessor.transform(t["new_obs"])
    random.seed(0)
    random.shuffle(data)
    split_point = max(len(data) - 5000, int(len(data) * 0.9))
    test_data = data[split_point:]
    data = data[:split_point]
    print("train batch size", len(data))
    print("test batch size", len(test_data))

    print("start training")
    for ix in range(1000):
        il_losses = []
        auto_losses = []
        ivd_losses = []
        fwd_losses = []
        for _ in range(len(data) // batch_size):
            batch = np.random.choice(data, batch_size)
            cur_fwd_loss, cur_ivd_loss, cur_il_loss, cur_auto_loss, _ = sess.run(
                [fwd_loss, ivd_loss, il_loss, autoencoder_loss, train_op],
                feed_dict={
                    observations: [t["encoded_obs"] for t in batch],
                    expert_actions: [t["action"] for t in batch],
                    orig_obs: [t["obs"] for t in batch],
                    next_obs: [t["encoded_next_obs"] for t in batch],
                })
            il_losses.append(cur_il_loss)
            auto_losses.append(cur_auto_loss)
            ivd_losses.append(cur_ivd_loss)
            fwd_losses.append(cur_fwd_loss)

        print("testing")
        test_ivd_losses = []
        test_il_losses = []
        test_auto_losses = []
        test_fwd_losses = []
        for _ in range(len(test_data) // batch_size):
            test_batch = np.random.choice(test_data, batch_size)
            test_fwd_loss, test_ivd_loss, test_il_loss, test_auto_loss = sess.run(
                [fwd_loss, ivd_loss, il_loss, autoencoder_loss],
                feed_dict={
                    observations: [t["encoded_obs"] for t in test_batch],
                    expert_actions: [t["action"] for t in test_batch],
                    orig_obs: [t["obs"] for t in test_batch],
                    next_obs: [t["encoded_next_obs"] for t in test_batch],
                })
            test_ivd_losses.append(test_ivd_loss)
            test_il_losses.append(test_il_loss)
            test_auto_losses.append(test_auto_loss)
            test_fwd_losses.append(test_fwd_loss)

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
            timesteps_total=ix,
            mean_loss=np.mean(il_losses) + np.mean(auto_losses) + np.mean(ivd_losses) + np.mean(fwd_losses),
            info={
                "train_il_acc": np.mean([np.exp(-l) for l in il_losses]),
                "train_auto_loss": np.mean(auto_losses),
                "train_fwd_loss": np.mean(fwd_losses),
                "train_ivd_acc": np.mean([np.exp(-l) for l in ivd_losses]),
                "test_il_mean_reward": np.mean(rewards),
                "test_il_acc": np.mean([np.exp(-l) for l in test_il_losses]),
                "test_auto_loss": np.mean(test_auto_losses),
                "test_fwd_loss": np.mean(test_fwd_losses),
                "test_ivd_acc": np.mean([np.exp(-l) for l in test_ivd_losses]),
            })

        if ix % 1 == 0:
            fname = "weights_{}".format(ix)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    register_trainable("pretrain", train)
    if args.image:
        run_experiments({
            "pretrain_{}".format(args.experiment): {
                "run": "pretrain",
                "config": {
                    "env_config": {
                        "background": args.background,
                    },
                    "data": os.path.expanduser(args.dataset),
                    "image": True,
                    "mode": grid_search(["ivd", "ivd_fwd"]),
                },
            }
        })
    else:
        run_experiments({
            "pretrain_{}".format(args.experiment): {
                "run": "pretrain",
                "config": {
                    "data": os.path.expanduser(args.dataset),
                    "mode": grid_search(["ivd", "fwd", "ivd_fwd", "il", "oracle"]),
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [256, 8],
                    },
                },
            }
        })
