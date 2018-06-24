from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import collections
from collections import deque
import json
import random

import pickle
import time
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
from ray.rllib.cartpole import ImageCartPole, CartpoleEncoder, parser, framestack_cartpole 
from ray.rllib.models.misc import normc_initializer
from ray.rllib.render_car import add_car_snow
from ray.rllib.models.preprocessors import NoPreprocessor
from ray.rllib.utils.atari_wrappers import wrap_deepmind, WarpFrame
from ray.rllib.utils.compression import unpack
from ray.tune import run_experiments, register_trainable, grid_search


# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)


GAN_STARTUP_ITERS = 3
SUCCESSOR_LOSS_WEIGHT = 1.0


def _scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
      scope in which the variables reside.
    trainable_only: bool
      whether or not to return only the variables that were marked as
      trainable.

    Returns
    -------
    vars: [tf.Variable]
      list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES
        if trainable_only else tf.GraphKeys.VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


cached_frames = []
deduplicated = 0
def decode_and_deduplicate(obj, snow_fn):
    global deduplicated
    if type(obj) in [str, unicode]:
        ret = unpack(obj)
    else:
        ret = obj
    if type(ret) is list:
        return ret  # not a framestacked image
    out = []
    for i in range(4):
        frame = ret[:, :, i:i+1]
        hit = False
        for original, rendered in cached_frames:
            if np.array_equal(frame, original):
                snowy_frame = rendered
                hit = True
                deduplicated += 1
        if not hit:
            snowy_frame = snow_fn(frame).astype("uint8")
            cached_frames.append((frame, snowy_frame))
            if len(cached_frames) > 8:
                cached_frames.pop(0)
        out.append(snowy_frame)
    return LazyFrames(out)


def save_image(data, name):
    dest = os.path.expanduser("~/Desktop/ae")
    if not os.path.exists(dest):
        os.makedirs(dest)
    imsave(os.path.join(dest, name), data)


def flatten(stacked_img):
    return stacked_img[:, :, -3:]


def _minimize_and_clip(optimizer, objective, variables, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, variables)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return gradients


def make_net(inputs, h_size, image, config, num_actions):
    if image:
        network = VisionNetwork(inputs, h_size, config.get("model", {}))
        feature_layer = network.outputs
        action_layer = slim.fully_connected(
            feature_layer, num_actions,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None, scope="action_layer_out")
    else:
        network = FullyConnectedNetwork(inputs, num_actions, config.get("model", {}))
        feature_layer = network.last_layer
        action_layer = network.outputs
    assert feature_layer.shape[1:] == (h_size,), feature_layer
    assert action_layer.shape[1:] == (num_actions,), action_layer
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
    if args.car:
        num_actions = 5
        num_options = 5
        prediction_frameskip = 1
        prediction_steps = 30
    else:
        num_actions = 2
        num_options = 2
        prediction_frameskip = 1
        prediction_steps = 10
    h_size = config["h_size"]
    data = config["data"]
    mode = config["mode"]
    env_config = config["env_config"]
    image = config.get("image", False)
    out_size = config.get("out_size", 200)
    batch_size = config.get("batch_size", 128)
    il_loss_enabled = mode == "il"
    ae_loss_enabled = mode in ["ae", "ae1step", "vae", "vae1step", "split_ae", "split_ae1step"]
    split_ae = mode in ["split_ae", "split_ae1step"]
    variational = ae_loss_enabled and mode.startswith("vae")
    ae_1step = mode in ["split_ae1step", "ae1step"]
    oracle_loss_enabled = mode == "oracle"
    ivd_loss_enabled = mode in ["ivd", "ivd_fwd"]
    forward_loss_enabled = mode in ["fwd", "ivd_fwd"]
    prediction_loss_enabled = mode in ["prediction", "split_ae", "split_ae1step", "successor_prediction"]
    successor_loss_enabled = mode in ["successor_prediction"]
    assert il_loss_enabled or oracle_loss_enabled or \
        ivd_loss_enabled or forward_loss_enabled or ae_loss_enabled or \
        prediction_loss_enabled
    gan_enabled = tf.placeholder(tf.float32, shape=())

    # Set up decoder network
    if image:
        observations = tf.placeholder(tf.float32, [None, 80, 80, k], name="observations")
    else:
        observations = tf.placeholder(tf.float32, [None, out_size], name="observations")
    with tf.variable_scope("embed_net"):
        feature_layer, action_layer = make_net(observations, h_size, image, config, num_actions)

    action_dist_cls = Categorical

    # Set up IL loss
    expert_options = tf.placeholder(tf.int32, [None], name="expert_options")
    expert_actions = tf.placeholder(tf.int32, [None], name="expert_actions")
    action_dist = action_dist_cls(action_layer)
    if il_loss_enabled:
        il_loss = -tf.reduce_mean(action_dist.logp(expert_actions))
    else:
        il_loss = tf.constant(0.0)
    act = action_dist.sample()
    print("IL loss", il_loss)

    # Set up reward prediction loss
    if args.car:
        pred_h0 = tf.concat([feature_layer, tf.one_hot(expert_options, 5)], axis=1)
    else:
        pred_h0 = tf.concat([feature_layer, tf.one_hot(expert_actions, 2)], axis=1)
    if not prediction_loss_enabled:
        pred_h0 = tf.stop_gradient(pred_h0)
    next_rewards = tf.placeholder(tf.float32, [None, prediction_steps], name="next_rewards")
    pred_h1 = slim.fully_connected(
        pred_h0, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="pred_h1")
    pred_out = slim.fully_connected(
        pred_h1, prediction_steps,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="reward_prediction")
    repeat = tf.placeholder(tf.int32, [None], name="repeat")
    # Only try to predict within repeat seqs
    can_predict = tf.expand_dims(
        tf.cast(tf.greater(repeat, prediction_steps * prediction_frameskip), tf.float32), 1)
    prediction_loss = tf.reduce_mean(
        tf.squared_difference(can_predict * pred_out, can_predict * next_rewards))

    # Set up oracle loss
    orig_obs = tf.placeholder(tf.float32, [None, 4], name="orig_obs")
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
        next_obs = tf.placeholder(tf.float32, [None, 80, 80, k], name="next_obs")
    else:
        next_obs = tf.placeholder(tf.float32, [None, out_size], name="next_obs")
    with tf.variable_scope("embed_net"):
        feature_layer2, _ = make_net(next_obs, h_size, image, config, num_actions)
    fused = tf.concat([feature_layer, feature_layer2], axis=1)
    if not ivd_loss_enabled:
        fused = tf.stop_gradient(fused)
    fused2 = slim.fully_connected(
        fused, 64,
        weights_initializer=normc_initializer(1.0),
        activation_fn=tf.nn.relu,
        scope="ivd_pred1")
    predicted_action = tf.squeeze(slim.fully_connected(
        fused2, num_actions,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="ivd_pred_out"))
    ivd_action_dist = action_dist_cls(predicted_action)
    ivd_loss = -tf.reduce_mean(
        ivd_action_dist.logp(expert_actions))
    print("Inv Dynamics loss", ivd_loss)

    # Setup successor loss
    if image:
        succ1 = tf.placeholder(tf.float32, [None, 80, 80, k], name="succ_obs1")
        succ2 = tf.placeholder(tf.float32, [None, 80, 80, k], name="succ_obs2")
        succ3 = tf.placeholder(tf.float32, [None, 80, 80, k], name="succ_obs3")
    else:
        succ1 = tf.placeholder(tf.float32, [None, out_size], name="succ_obs1")
        succ2 = tf.placeholder(tf.float32, [None, out_size], name="succ_obs2")
        succ3 = tf.placeholder(tf.float32, [None, out_size], name="succ_obs3")
    with tf.variable_scope("target_net"):
        succ1_target, _ = make_net(succ1, h_size, image, config, num_actions)
        succ2_target, _ = make_net(succ2, h_size, image, config, num_actions)
        succ3_target, _ = make_net(succ3, h_size, image, config, num_actions)
    if successor_loss_enabled:
        f_in = pred_h0
    else:
        f_in = tf.stop_gradient(pred_h0)
    pred_succ1 = slim.fully_connected(
        f_in, h_size,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="pred_succ1")
    pred_succ2 = slim.fully_connected(
        f_in, h_size,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="pred_succ2")
    pred_succ3 = slim.fully_connected(
        f_in, h_size,
        weights_initializer=normc_initializer(0.01),
        activation_fn=None, scope="pred_succ3")
    mean_val = tf.reduce_mean(tf.square(succ1_target))
    successor_loss1 = SUCCESSOR_LOSS_WEIGHT * tf.reduce_mean(
        tf.squared_difference(pred_succ1, succ1_target) / mean_val)
    successor_loss2 = SUCCESSOR_LOSS_WEIGHT * tf.reduce_mean(
        tf.squared_difference(pred_succ2, succ2_target) / mean_val)
    successor_loss3 = SUCCESSOR_LOSS_WEIGHT * tf.reduce_mean(
        tf.squared_difference(pred_succ3, succ3_target) / mean_val)

    update_target_expr = []
    embed_vars = _scope_vars("embed_net")
    target_embed_vars = _scope_vars("target_net")
    print("Embed vars", embed_vars)
    print("Target embed vars", target_embed_vars)
    for var, var_target in zip(
        sorted(embed_vars, key=lambda v: v.name),
            sorted(target_embed_vars, key=lambda v: v.name)):
        update_target_expr.append(var_target.assign(var))
    print("Update target ops", update_target_expr)
    update_target_expr = tf.group(*update_target_expr)

    # Set up forward loss
    if args.car:
        feature_and_action = tf.concat(
            [feature_layer, tf.one_hot(expert_actions, 4)], axis=1)
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
        no_snow_latent_vector = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    else:
        no_snow_latent_vector = feature_and_action

    if split_ae:
        with tf.variable_scope("snow_net"):
            snow_latent_vector, _ = make_net(
                observations, 512, image, config, num_actions)

    neg_regressor_loss = tf.constant(0.0)
    pos_regressor_loss = tf.constant(0.0)
    if ae_loss_enabled:
        if split_ae:
            ae_no_snow_out = decode_image(no_snow_latent_vector, 1)
            with tf.variable_scope("snow_out"):
                ae_snow_out = decode_image(snow_latent_vector, 1)
            with tf.variable_scope("snow_out_regressor"):
                _, regressor_out = make_net(ae_snow_out, h_size, image, config, prediction_steps)
            pos_regressor_loss = tf.reduce_mean(
                tf.squared_difference(can_predict * regressor_out, can_predict * next_rewards)) * gan_enabled
            neg_regressor_loss = - pos_regressor_loss
            # TODO(ekl) set up gan style training on regressor
            autoencoder_out = tf.maximum(ae_snow_out, ae_no_snow_out * gan_enabled)
        else:
            autoencoder_out = decode_image(no_snow_latent_vector, 1)
            ae_snow_out = autoencoder_out
            ae_no_snow_out = autoencoder_out
    else:
        # still try to reproduce the image, but don't optimize prior layers
        autoencoder_out = decode_image(tf.stop_gradient(no_snow_latent_vector), 1)
        ae_snow_out = autoencoder_out
        ae_no_snow_out = autoencoder_out

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
        ae_loss = tf.reduce_mean(tf.squared_difference(target, autoencoder_out)) / 100
    print("(v)ae loss", ae_loss)

    # Set up optimizer
    optimizer = tf.train.AdamOptimizer()
    regressor_optimizer = tf.train.AdamOptimizer()
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    regressor_vars = [v for v in variables if "regressor" in v.name]
    non_regressor_vars = [v for v in variables if "regressor" not in v.name]
    summed_loss = (
        oracle_loss + il_loss + ivd_loss + ae_loss + prediction_loss +
        neg_regressor_loss + successor_loss1 + successor_loss2 + successor_loss3 +
        fwd_loss * config.get("fwd_weight", 0.0))
    grads = _minimize_and_clip(optimizer, summed_loss, non_regressor_vars)
    train_op = optimizer.apply_gradients(grads)
    if split_ae:
        regressor_train_op = regressor_optimizer.minimize(pos_regressor_loss, var_list=regressor_vars)
    else:
        regressor_train_op = tf.constant(0.0)

    if args.car:
        env = gym.make("CarRacing-v0")
    else:
        env = gym.make("CartPole-v0")
    snow_fn = lambda obs: obs
    if args.car:
        snow_fn = lambda obs: add_car_snow(
            obs, env_config["num_snow"], env_config["background"] == "noise")
        env = wrap_deepmind(env, snow_fn=snow_fn)
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
    detected_gpus = ray.services._autodetect_num_gpus()
    if detected_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            str(random.choice(range(detected_gpus))))
    print("CUDA: " + os.environ.get("CUDA_VISIBLE_DEVICES"))
    sess = tf.Session(config=tf_config)
    print("Created session")
    sess.run(tf.global_variables_initializer())
    print("Initialized variables")

    vars = TensorFlowVariables(summed_loss, sess)

    def get_next_rewards(data, start_i):
        rew = [0] * prediction_steps * prediction_frameskip
        for i in range(prediction_steps * prediction_frameskip):
            offset = start_i + i
            if offset < len(data):
                if i > 0:
                    prev = rew[i-1]
                else:
                    prev = 0
                rew[i] = prev + data[offset]["reward"]
                if data[offset]["done"]:
                    break
        res = rew[prediction_frameskip-1::prediction_frameskip]
        assert len(res) == prediction_steps
        return res

    def get_future_obs(data, start_i, offset):
        future_i = start_i
        for i in range(start_i, future_i + 1):
            if i >= len(data) or data[i]["done"]:
                return None  # end of rollout
        return data[future_i]["encoded_obs"]

    print("Loading data", env_config)
    matches = glob.glob(data)
    data = []
    for fdata in matches:
        for i, x in enumerate(open(fdata).readlines()):
            if i % 10000 == 0:
                print("Loading data tuple", i)
            try:
                t = json.loads(x)
                t["obs"] = decode_and_deduplicate(t["obs"], snow_fn)
                t["new_obs"] = decode_and_deduplicate(t["new_obs"], snow_fn)
                if "option" not in t:
                    t["option"] = 1
                data.append(t)
            except Exception as e:
                print(e)
    print("preprocessing data", len(data))
    for i, t in enumerate(data):
        t["next_rewards"] = get_next_rewards(data, i)
    print("num deduplicated", deduplicated)

    if args.car:
        data_out = []
        for t in data:
            t["encoded_obs"] = t["obs"]
            t["encoded_next_obs"] = t["new_obs"]
            t["obs"] = [0, 0, 0, 0]  # "true" latent state not available
            t["option"] = t["option"] - 1  # zero index it
            if t["action"] == 100:  # legacy idle action
                t["action"] = 4
            if len(data_out) % 10000 == 0:
                print("Loaded frames", len(data_out))
            data_out.append(t)
        data = data_out
    elif args.image:
        data = framestack_cartpole(data, k, config["env_config"], args)
    else:
        for t in data:
            t["encoded_obs"] = preprocessor.transform(t["obs"])
            t["encoded_next_obs"] = preprocessor.transform(t["new_obs"])

    # do this after the first pass to share the decoded arrays
    zero_obs = np.zeros_like(np.array(data[0]["encoded_obs"]))
    for i, t in enumerate(data):
        # Successor obs
        succ1_obs = get_future_obs(data, i, 1)
        succ2_obs = get_future_obs(data, i, (prediction_frameskip * prediction_steps) // 5)
        succ3_obs = get_future_obs(data, i, (prediction_frameskip * prediction_steps) // 2)
        if succ1_obs is None:
            t["succ1"] = zero_obs
        else:
            t["succ1"] = succ1_obs
        if succ2_obs is None:
            t["succ2"] = zero_obs
        else:
            t["succ2"] = succ2_obs
        if succ3_obs is None:
            t["succ3"] = zero_obs
        else:
            t["succ3"] = succ3_obs

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
        ("prediction", prediction_loss),
        ("neg_regressor_loss", neg_regressor_loss),
        ("succ1", successor_loss1),
        ("succ2", successor_loss2),
        ("succ3", successor_loss3),
    ]

    print("start training")
    miniepoch_size = min(100000, len(data))
    num_miniepochs = int(np.ceil(len(data) / 100000.0))
    for ix in range(1000):
        print("start epoch", ix)
        for i2x in range(num_miniepochs):
            it = ix * num_miniepochs + i2x
            print("start miniepoch", it)
            train_losses = collections.defaultdict(list)
            sample_time = 0.0
            run_time = 0.0
            regressor_time = 0.0
            for _ in range(miniepoch_size // batch_size):
                start = time.time()
                batch = random.sample(data, batch_size)
                sample_time += time.time() - start
                start = time.time()
                results = sess.run(
                    [tensor for (_, tensor) in LOSSES] + [train_op],
                    feed_dict={
                        observations: np.array([t["encoded_obs"] for t in batch]),
                        expert_actions: [t["action"] for t in batch],
                        expert_options: [t["option"] for t in batch],
                        orig_obs: [t["obs"] for t in batch],
                        next_obs: np.array([t["encoded_next_obs"] for t in batch]),
                        succ1: np.array([t["succ1"] for t in batch]),
                        succ2: np.array([t["succ2"] for t in batch]),
                        succ3: np.array([t["succ3"] for t in batch]),
                        next_rewards: [t["next_rewards"] for t in batch],
                        repeat: [t.get("repeat", 0) for t in batch],
                        gan_enabled: it > GAN_STARTUP_ITERS,
                    })
                run_time += time.time() - start
                for (name, _), value in zip(LOSSES, results):
                    train_losses[name].append(value)
                start = time.time()
                if split_ae:
                    for _ in range(10):
                        sess.run(
                            regressor_train_op,
                            feed_dict={
                                observations: np.array([t["encoded_obs"] for t in batch]),
                                expert_actions: [t["action"] for t in batch],
                                expert_options: [t["option"] for t in batch],
                                orig_obs: [t["obs"] for t in batch],
                                next_obs: np.array([t["encoded_next_obs"] for t in batch]),
                                succ1: np.array([t["succ1"] for t in batch]),
                                succ2: np.array([t["succ2"] for t in batch]),
                                succ3: np.array([t["succ3"] for t in batch]),
                                next_rewards: [t["next_rewards"] for t in batch],
                                repeat: [t.get("repeat", 0) for t in batch],
                                gan_enabled: True,
                            })
                regressor_time += time.time() - start
            print("sample time", sample_time, "run time", run_time, "regressor time", regressor_time)

            print("testing miniepoch", it)
            test_losses = collections.defaultdict(list)
            for jx in range(max(1, len(test_data) // batch_size)):
                test_batch = np.random.choice(test_data, batch_size)
                results = sess.run(
                    [tensor for (_, tensor) in LOSSES] + [autoencoder_out, ae_snow_out, ae_no_snow_out],
                    feed_dict={
                        observations: np.array([t["encoded_obs"] for t in test_batch]),
                        expert_actions: [t["action"] for t in test_batch],
                        expert_options: [t["option"] for t in test_batch],
                        orig_obs: [t["obs"] for t in test_batch],
                        next_obs: np.array([t["encoded_next_obs"] for t in test_batch]),
                        succ1: np.array([t["succ1"] for t in test_batch]),
                        succ2: np.array([t["succ2"] for t in test_batch]),
                        succ3: np.array([t["succ3"] for t in test_batch]),
                        next_rewards: [t["next_rewards"] for t in test_batch],
                        repeat: [t.get("repeat", 0) for t in test_batch],
                        gan_enabled: 1,
                    })
                for (name, _), value in zip(LOSSES, results):
                    test_losses[name].append(value)
                if jx <= 5:
                    save_image(flatten(np.array(test_batch[0]["encoded_obs"])), "{}_{}_{}_in.png".format(mode, ix, jx))
                    save_image(results[-3][0].squeeze(), "{}_{}_{}_out.png".format(mode, ix, jx))
                    if split_ae:
                        save_image(results[-1][0].squeeze(), "{}_{}_{}_out_feat.png".format(mode, ix, jx))
                        save_image(results[-2][0].squeeze(), "{}_{}_{}_out_noise.png".format(mode, ix, jx))

            loss_info = {
                "epoch": ix,
                "miniepoch": it,
            }
            mean_train_loss = 0.0
            for name, values in train_losses.items():
                mean_train_loss += np.mean(values)
                loss_info["train_{}_loss".format(name)] = np.mean(values)
            for name, values in test_losses.items():
                loss_info["test_{}_loss".format(name)] = np.mean(values)

            reporter(
                timesteps_total=it, mean_loss=mean_train_loss, info=loss_info)

            print("updating target network")
            sess.run(update_target_expr)

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
                        "num_snow": grid_search(
                            [int(x) for x in args.grid_snow.split(",")]
                            if args.grid_snow else [args.num_snow])
                    },
                    "data": os.path.expanduser(args.dataset),
                    "h_size": args.h_size,
                    "image": True,
                    "mode": grid_search(
                        args.pretrain_mode.split(",")
                        if args.pretrain_mode else ["il"]),
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
                        "num_snow": args.num_snow
                    },
                    "data": os.path.expanduser(args.dataset),
                    "h_size": args.h_size,
                    "image": True,
                    "mode": grid_search(
                        args.pretrain_mode.split(",")
                        if args.pretrain_mode else ["il"]),
                },
            }
        })
    else:
        run_experiments({
            "pretrain_{}".format(args.experiment): {
                "run": "pretrain",
                "config": {
                    "data": os.path.expanduser(args.dataset),
                    "h_size": args.h_size,
                    "mode": grid_search(["ivd", "fwd", "ivd_fwd", "il", "oracle"]),
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [256, 8],
                    },
                },
            }
        })
