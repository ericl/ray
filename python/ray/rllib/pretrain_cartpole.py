from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
from collections import deque
import json
import random

import pickle
import numpy as np
import gym
from scipy.misc import imsave
import tensorflow as tf
import tensorflow.contrib.slim as slim

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.action_dist import Categorical, Deterministic
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.models.visionnet import VisionNetwork
from ray.rllib.cartpole import ImageCartPole, CartpoleEncoder, parser
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.render_cartpole import render_frame
from ray.rllib.utils.atari_wrappers import wrap_deepmind, WarpFrame
from ray.rllib.utils.compression import unpack
from ray.tune import run_experiments, register_trainable, grid_search


def decode(obj):
    if type(obj) in [str, unicode]:
        return unpack(obj)
    else:
        return obj


def save_image(data, name):
    dest = os.path.expanduser("~/Desktop/ae")
    if not os.path.exists(dest):
        os.makedirs(dest)
    imsave(os.path.join(dest, name), data)


def flatten(stacked_img):
    return stacked_img[:, :, -3:]


def _minimize_and_clip(optimizer, objective, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return gradients


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


def decode_image(feature_layer, k):
    print("Feature layer", feature_layer)
    expanded = tf.expand_dims(tf.expand_dims(feature_layer, 1), 1)
    print("expanded", expanded)
    fc2_inv = slim.conv2d_transpose(
        expanded, 512, [1, 1], activation_fn=None, scope="fc2_inv")
    print("fc2_inv", fc2_inv)
    fc1_inv = slim.conv2d_transpose(
        fc2_inv, 32, [10, 10], 1, padding="VALID", scope="fc1_inv")
    print("fc1_inv", fc1_inv)
    conv2_inv = slim.conv2d_transpose(
        fc1_inv, 16, [4, 4], 2, scope="conv2_inv")
    print("conv2_inv", conv2_inv)
    conv1_inv = slim.conv2d_transpose(
        conv2_inv, k, [8, 8], 4, scope="conv1_inv")
    print("conv1_inv", conv1_inv)
    return conv1_inv


def train(config, reporter):
    k = 4
    h_size = config["h_size"]
    data = config["data"]
    mode = config["mode"]
    env_config = config["env_config"]
    image = config.get("image", False)
    out_size = config.get("out_size", 200)
    batch_size = config.get("batch_size", 128)
    il_loss_enabled = mode == "il"
    ae_loss_enabled = mode in ["ae", "ae1step", "vae", "vae1step"]
    variational = ae_loss_enabled and mode.startswith("vae")
    ae_1step = mode == "ae1step"
    oracle_loss_enabled = mode == "oracle"
    ivd_loss_enabled = mode in ["ivd", "ivd_fwd"]
    forward_loss_enabled = mode in ["fwd", "ivd_fwd"]
    assert il_loss_enabled or oracle_loss_enabled or \
        ivd_loss_enabled or forward_loss_enabled or ae_loss_enabled

    # Set up decoder network
    if image:
        observations = tf.placeholder(tf.float32, [None, 80, 80, k])
    else:
        observations = tf.placeholder(tf.float32, [None, out_size])
    feature_layer, action_layer = make_net(observations, h_size, image, config)

    if args.car:
        action_dist_cls = Deterministic
    else:
        action_dist_cls = Categorical

    # Set up IL loss
    if args.car:
        expert_actions = tf.placeholder(tf.float32, [None])
        action_dist = action_dist_cls(action_layer)
    else:
        expert_actions = tf.placeholder(tf.int32, [None])
        action_dist = action_dist_cls(action_layer)
    if il_loss_enabled:
        il_loss = -tf.reduce_mean(action_dist.logp(expert_actions))
    else:
        il_loss = tf.constant(0.0)
    act = action_dist.sample()
    print("IL loss", il_loss)

    # Set up oracle loss
    if args.car:
        orig_obs = tf.placeholder(tf.float32, [None, 80, 80, k])
    else:
        orig_obs = tf.placeholder(tf.float32, [None, 4])
    oracle_in = feature_layer
    if oracle_loss_enabled:
        assert not args.car, "Not supported"
        recons_obs = slim.fully_connected(
            oracle_in, 4,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None, scope="fc_oracle_out")
        oracle_loss = tf.reduce_mean(tf.squared_difference(orig_obs, recons_obs))
    else:
        oracle_loss = tf.constant(0.0)
    print("oracle loss", oracle_loss)

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
        fused, 1 if args.car else 2,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="ivd_pred_out").squeeze()
    ivd_action_dist = action_dist_cls(predicted_action)
    if ivd_loss_enabled:
        ivd_loss = -tf.reduce_mean(
            ivd_action_dist.logp(expert_actions))
    else:
        ivd_loss = tf.constant(0.0)
    print("Inv Dynamics loss", ivd_loss)

    # Set up forward loss
    if args.car:
        feature_and_action = tf.concat(
            [feature_layer, tf.expand_dims(expert_actions, 1)], axis=1)
    else:
        feature_and_action = tf.concat(
            [feature_layer, tf.one_hot(expert_actions, 2)], axis=1)
    fwd1 = slim.fully_connected(
        feature_and_action, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="fwd1")
    fwd2 = slim.fully_connected(
        fwd1, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="fwd2")
    fwd_delta = slim.fully_connected(
        fwd2, h_size,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="fwd_delta")
    if forward_loss_enabled:
        fwd_out = tf.add(tf.stop_gradient(feature_layer), fwd_delta)
        fwd_loss = tf.reduce_mean(
            tf.squared_difference(tf.stop_gradient(feature_layer2), fwd_out)
        )
    else:
        fwd_loss = tf.constant(0.0)

    # Set up autoencoder loss
    if variational:
        mu = slim.fully_connected(
            feature_and_action, h_size,
            weights_initializer=normc_initializer(1.0),
            activation_fn=tf.nn.relu, scope="vae_mean")
        sigma = slim.fully_connected(
            feature_and_action, h_size,
            weights_initializer=normc_initializer(1.0),
            activation_fn=tf.nn.relu, scope="vae_stddev")
        latent_vector = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    else:
        latent_vector = feature_and_action

    if ae_loss_enabled:
        autoencoder_out = decode_image(latent_vector, 1)
    else:
        # still try to reproduce the image, but don't optimize prior layers
        autoencoder_out = decode_image(tf.stop_gradient(latent_vector), 1)

    if ae_1step:
        target = next_obs[..., -1:]
    else:
        target = observations[..., -1:]
    if variational:
        generation_loss = tf.reduce_mean(tf.squared_difference(target, autoencoder_out))
        kl_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1))
        ae_loss = generation_loss + kl_loss
    else:
        ae_loss = tf.reduce_mean(tf.squared_difference(target, autoencoder_out))
    print("(v)ae loss", ae_loss)

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer()
    summed_loss = (
        oracle_loss + il_loss + ivd_loss + ae_loss +
        fwd_loss * config.get("fwd_weight", 0.0))
    grads = _minimize_and_clip(optimizer, summed_loss)
    train_op = optimizer.apply_gradients(grads)

    if args.car:
        env = gym.make("CarRacing-v0")
    else:
        env = gym.make("CartPole-v0")
    if args.car:
        resizer = WarpFrame(env, 80)
        env = wrap_deepmind(env)
        preprocessor = NoPreprocessor(env.observation_space, {})
    elif args.image:
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
    for t in data:
        t["obs"] = decode(t["obs"])
        t["new_obs"] = decode(t["new_obs"])

    def render(raw_obs, config):
        if args.car:
            return resizer.observation(raw_obs)
        else:
            return render_frame(raw_obs, config)

    if args.image or args.car:
        frames = deque([], maxlen=k)
        data_out = []
        for t in data:
            ok = len(frames) >= k
            if len(frames) == 0:
                frames.append(render(t["obs"], config["env_config"]))
            if ok:
                t["encoded_obs"] = np.concatenate(frames, axis=2)
            frames.append(render(t["new_obs"], config["env_config"]))
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

    LOSSES = [
        ("ae", ae_loss),
        ("fwd", fwd_loss),
        ("il", il_loss),
        ("ivd", ivd_loss),
        ("oracle", oracle_loss),
    ]

    print("start training")
    for ix in range(1000):
        train_losses = collections.defaultdict(list)
        for _ in range(len(data) // batch_size):
            batch = np.random.choice(data, batch_size)
            results = sess.run(
                [tensor for (_, tensor) in LOSSES] + [train_op],
                feed_dict={
                    observations: [t["encoded_obs"] for t in batch],
                    expert_actions: [t["action"] for t in batch],
                    orig_obs: [t["obs"] for t in batch],
                    next_obs: [t["encoded_next_obs"] for t in batch],
                })
            for (name, _), value in zip(LOSSES, results):
                train_losses[name].append(value)

        print("testing")
        test_losses = collections.defaultdict(list)
        for jx in range(max(1, len(test_data) // batch_size)):
            test_batch = np.random.choice(test_data, batch_size)
            results = sess.run(
                [tensor for (_, tensor) in LOSSES] + [autoencoder_out],
                feed_dict={
                    observations: [t["encoded_obs"] for t in test_batch],
                    expert_actions: [t["action"] for t in test_batch],
                    orig_obs: [t["obs"] for t in test_batch],
                    next_obs: [t["encoded_next_obs"] for t in test_batch],
                })
            for (name, _), value in zip(LOSSES, results):
                test_losses[name].append(value)
            if jx <= 5:
                save_image(flatten(test_batch[0]["encoded_obs"]), "{}_{}_{}_in.png".format(mode, ix, jx))
                save_image(results[-1][0].squeeze(), "{}_{}_{}_out.png".format(mode, ix, jx))

        # Evaluate IL performance
        rewards = []
        if not args.car:  # TODO
            for _ in range(100):
                obs = env.reset()
                reward = 0
                done = False
                while not done:
                    action = sess.run(act, feed_dict={observations: [preprocessor.transform(obs)]})[0]
                    obs, rew, done, _ = env.step(action)
                    reward += rew
                rewards.append(reward)

        loss_info = {}
        mean_train_loss = 0.0
        for name, values in train_losses.items():
            mean_train_loss += np.mean(values)
            loss_info["train_{}_loss".format(name)] = np.mean(values)
        for name, values in test_losses.items():
            loss_info["test_{}_loss".format(name)] = np.mean(values)

        reporter(
            timesteps_total=ix, mean_loss=mean_train_loss, info=loss_info)

        if ix % 1 == 0:
            fname = "weights_{}".format(ix)
            with open(fname, "wb") as f:
                f.write(pickle.dumps(vars.get_weights()))
                print("Saved weights to " + fname)


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    register_trainable("pretrain", train)
    if args.car:
        run_experiments({
            "pretrain_car_{}".format(args.experiment): {
                "run": "pretrain",
                "config": {
                    "env_config": {
                        "background": args.background,
                    },
                    "data": os.path.expanduser(args.dataset),
                    "h_size": 32,
                    "image": True,
                    "mode": grid_search(["ae", "ae1step", "vae", "ivd"]),
                },
            }
        })
    elif args.image:
        run_experiments({
            "pretrain_{}".format(args.experiment): {
                "run": "pretrain",
                "config": {
                    "env_config": {
                        "background": args.background,
                    },
                    "data": os.path.expanduser(args.dataset),
                    "h_size": 8,
                    "image": True,
                    "mode": grid_search(["vae", "vae1step"]),
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
