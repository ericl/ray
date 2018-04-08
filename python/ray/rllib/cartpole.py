from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import deque
import math
import numpy as np
import pickle
import gym
from gym import spaces
import random
import scipy.stats
import time
import tensorflow as tf

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.models.visionnet import VisionNetwork
from ray.tune import run_experiments, grid_search, register_env
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent
#from ray.rllib import bullet_cartpole
from ray.rllib.render_cartpole import render_frame


parser = argparse.ArgumentParser()
parser.add_argument("--car", action="store_true")
parser.add_argument("--image", action="store_true")
parser.add_argument("--decode-model", default=None)
parser.add_argument("--background", default="noise")
parser.add_argument("--experiment", default="cartpole-decode")
parser.add_argument("--dataset", default=None)
parser.add_argument("--h-size", default=8, type=int)


def load_image_model(weights_file, obs_ph, sess, h_size):
    model_config = {}
    network = VisionNetwork(obs_ph, h_size, model_config)
    vars = TensorFlowVariables(network.outputs, sess)
    sess.run(tf.global_variables_initializer())
    with open(weights_file, "rb") as f:
        vars.set_weights(pickle.loads(f.read()))
    return network


def load_model(weights_file, obs_ph, sess):
    model_config = {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [256, 8],
    }
    network = FullyConnectedNetwork(obs_ph, 2, model_config)
    vars = TensorFlowVariables(network.outputs, sess)
    sess.run(tf.global_variables_initializer())
    with open(weights_file, "rb") as f:
        vars.set_weights(pickle.loads(f.read()))
    return network


def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


class Flatten(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape
        prod = 1
        for n in self.shape:
            prod *= n
        self.shape = (prod,)

    def transform(self, obs):
        return obs.flatten()


class ImageCartPole(gym.Wrapper):
    def __init__(self, env, k, env_config):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.env_config = env_config
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = (80, 80, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def reset(self):
        ob = render_frame(self.env.reset(), self.env_config)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = render_frame(ob, self.env_config)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class Space(object):
    def __init__(self, shape):
        self.shape = shape


class ImageDecoder(Preprocessor):
    def _init(self):
        self.decode_model = self._options["custom_options"].get("decode_model")
        self.sess = tf.Session()
        self.obs_ph = tf.placeholder(
            tf.float32, [None] + list(self._obs_space.shape))
        self.h_size = self._options["custom_options"]["h_size"]
        print("Image decoder input: " + str(self.obs_ph))
        self.decoder = load_image_model(self.decode_model, self.obs_ph, self.sess, self.h_size)
        self.shape = (self.h_size,)

    def transform(self, obs):
        out = self.sess.run(self.decoder.outputs, feed_dict={
            self.obs_ph: [obs]
        })[0]
        return out


class Decoder(Preprocessor):
    def _init(self):
        self.decode_model = self._options["custom_options"].get("decode_model")
        self.sess = tf.Session()
        self.obs_ph = tf.placeholder(
            tf.float32, [None] + list(self._obs_space.shape))
        print("Decoder input: " + str(self.obs_ph))
        self.decoder = load_model(self.decode_model, self.obs_ph, self.sess)
        self.shape = (8,)

    def transform(self, obs):
        out = self.sess.run(self.decoder.last_layer, feed_dict={
            self.obs_ph: [obs]
        })[0]
        return out


class CartpoleEncoder(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape
        np.random.seed(self._options["custom_options"]["seed"])

        out_size = self._options["custom_options"]["out_size"]
        assert out_size % 4 == 0
        self.elem_size = out_size // 4

        self.decode_model = self._options["custom_options"].get("decode_model")
        if self.decode_model:
            self.decoder = Decoder(Space((out_size,)), self._options)
            self.shape = self.decoder.shape
        else:
            self.shape = (out_size,)

    def transform(self, obs):
        W = [1.0, 3.0, 0.30, 3.0]  # widths
        out = []
        means = []
        for i, width in enumerate(W):
            half = self.elem_size / 2
            mean = half * obs[i] / width + half
            means.append(mean)
            std = 1
            elem = [
                normpdf(j, mean, std) + math.sin(j + mean) #+ random.random()
                for j in range(self.elem_size)
            ]
            out.extend(elem)
        if self.decode_model:
            out = self.decoder.transform(out)
        return out


if __name__ == '__main__':
    ModelCatalog.register_custom_preprocessor("encode_pseudoimg", CartpoleEncoder)
    ModelCatalog.register_custom_preprocessor("flatten", Flatten)
    ModelCatalog.register_custom_preprocessor("decoder", Decoder)
    ModelCatalog.register_custom_preprocessor("img_decoder", ImageDecoder)
    register_env(
        "ImageCartPole-v0",
        lambda env_config: ImageCartPole(gym.make("CartPole-v0"), 4, env_config))
#    register_env(
#        "BulletCartPole-v0",
#        lambda config: bullet_cartpole.BulletCartpole(
#            bullet_cartpole.add_opts(
#                argparse.ArgumentParser()).parse_args([]),
#            discrete_actions=True))

    ray.init()
    args = parser.parse_args()

    decode_model = args.decode_model and os.path.expanduser(args.decode_model)
    if args.image or args.car:
        if decode_model:
            model_opts = {
                "custom_preprocessor": "img_decoder",
                "custom_options": {
                    "decode_model": decode_model,
                    "h_size": args.h_size,
                },
            }
        else:
            model_opts = {}
        if args.car:
            run_experiments({
                args.experiment: {
                    "run": "PPO",
                    "env": "CarRacing-v0",
                    "repeat": 1,
                    "trial_resources": {
                        "cpu": 1,
                        "gpu": 1,
                        "extra_cpu": lambda spec: spec.config.num_workers,
                    },
                    "config": {
                        "gamma": 0.95,
                        "devices": ["/cpu:0"],
                        "num_sgd_iter": 10,
                        "num_workers": 7,
                        "model": model_opts,
                    },
                }
            })
        else:
            run_experiments({
                args.experiment: {
                    "run": "PPO",
                    "env": "ImageCartPole-v0",
                    "repeat": 1,
                    "trial_resources": {
                        "cpu": 1,
                        "gpu": 1,
                        "extra_cpu": lambda spec: spec.config.num_workers,
                    },
                    "stop": {
                        "episode_reward_mean": 200,
                    },
                    "config": {
                        "env_config": {
                            "background": args.background,
                        },
                        "devices": ["/gpu:0"],
                        "num_sgd_iter": 10,
                        "num_workers": 7,
                        "model": model_opts,
                    },
                }
            })
    else:
        run_experiments({
            args.experiment: {
                "run": "PPO",
                "env": "CartPole-v0",
                "repeat": 1,
                "trial_resources": {
                    "cpu": 1,
                    "extra_cpu": lambda spec: spec.config.num_workers,
                },
                "stop": {
                    "episode_reward_mean": 200,
                },
                "config": {
                    "num_sgd_iter": 10,
                    "num_workers": 1,
                    "model": {
                        "custom_preprocessor": "encode_pseudoimg",
                        "custom_options": {
                            "seed": 0,
                            "out_size": grid_search([200]),
                            "decode_model": decode_model,
                        },
                    },
                },
            }
        })
