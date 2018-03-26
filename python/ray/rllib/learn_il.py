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


def train(config, reporter):
    k = 4
    data = config["data"]
    mode = config["mode"]
    image = config.get("image", False)
    out_size = config.get("out_size", 200)
    batch_size = config.get("batch_size", 128)
    il_loss_enabled = mode == "il"
    autoencoder_loss_enabled = mode == "oracle"
    inv_dyn_loss_enabled = mode == "ivd"
    assert il_loss_enabled or autoencoder_loss_enabled or inv_dyn_loss_enabled

    # Set up decoder network
    if image:
        observations = tf.placeholder(tf.float32, [None, 42, 42, k])
        network = VisionNetwork(observations, 2, config.get("model", {}))
    else:
        observations = tf.placeholder(tf.float32, [None, out_size])
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
    if image:
        next_obs = tf.placeholder(tf.float32, [None, 42, 42, k])
        network2 = VisionNetwork(next_obs, 2, config.get("model", {}))
    else:
        next_obs = tf.placeholder(tf.float32, [None, out_size])
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
    if args.image:
        env = ImageCartPole(env, k)
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
                frames.append(render_frame(t["obs"]))
            if ok:
                t["encoded_obs"] = np.concatenate(frames, axis=2)
            frames.append(render_frame(t["new_obs"]))
            if ok:
                t["encoded_next_obs"] = np.concatenate(frames, axis=2)
                data_out.append(t)
            if t["done"]:
                frames.clear()
        data = data_out
    else:
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
    for ix in range(1000):
        il_losses = []
        auto_losses = []
        inv_dyn_losses = []
        errors = [[], [], [], []]
        for _ in range(len(data) // batch_size):
            batch = np.random.choice(data, batch_size)
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
            timesteps_total=ix, mean_loss=np.mean(il_losses) + auto_loss + ivd_loss, info={
                "decoder_reconstruction_error": {
                    "cart_pos": errors[0] / means[0],
                    "cart_velocity": errors[1] / means[1],
                    "pole_angle": errors[2] / means[2],
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

        if ix % 1 == 0:
            fname = "weights_{}".format(ix)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    register_trainable("il", train)
    if args.image:
        run_experiments({
            "iltrain_image": {
                "run": "il",
                "config": {
                    "data": os.path.expanduser("~/Desktop/cartpole-expert.json"),
                    "image": True,
                    "mode": grid_search(["il", "ivd", "oracle"]),
                    "model": {
                        "conv_filters": [
                            [16, [4, 4], 2],
                            [32, [4, 4], 2],
                            [512, [11, 11], 1],
                        ],
                    },
                },
            }
        })
    else:
        run_experiments({
            "iltrain": {
                "run": "il",
                "config": {
                    "data": os.path.expanduser("~/Desktop/cartpole-expert.json"),
                    "mode": grid_search(["il", "ivd", "oracle"]),
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [256, 8],
                    },
                },
            }
        })
