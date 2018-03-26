from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
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
from ray.tune import run_experiments, grid_search, register_env
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent
#from ray.rllib import bullet_cartpole
from ray.rllib.render_cartpole import render_frame


parser = argparse.ArgumentParser()
parser.add_argument("--image", action="store_true")


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
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = (80, 80, 1)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def reset(self):
        ob = render_frame(self.env.reset())
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = render_frame(ob)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)


class CartpoleEncoder(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape
        np.random.seed(self._options["custom_options"]["seed"])

        out_size = self._options["custom_options"]["out_size"]
        assert out_size % 4 == 0
        self.elem_size = out_size // 4

        self.decode_model = self._options["custom_options"].get("decode_model")
        if self.decode_model:
            self.sess = tf.Session()
            self.obs_ph = tf.placeholder(
                tf.float32, [None, out_size])
            self.decoder = load_model(self.decode_model, self.obs_ph, self.sess)
            self.shape = (8,)
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
            out = self.sess.run(self.decoder.last_layer, feed_dict={
                self.obs_ph: [out]
            })[0]
        return out


if __name__ == '__main__':
    ModelCatalog.register_custom_preprocessor("encode_pseudoimg", CartpoleEncoder)
    ModelCatalog.register_custom_preprocessor("flatten", Flatten)
    register_env(
        "ImageCartPole-v0",
        lambda config: ImageCartPole(gym.make("CartPole-v0"), 2))
#    register_env(
#        "BulletCartPole-v0",
#        lambda config: bullet_cartpole.BulletCartpole(
#            bullet_cartpole.add_opts(
#                argparse.ArgumentParser()).parse_args([]),
#            discrete_actions=True))

    ray.init()
    args = parser.parse_args()

    if args.image:
        run_experiments({
            "image-cartpole": {
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
                    "devices": ["/gpu:1"],
                    "num_sgd_iter": 10,
                    "num_workers": 1,
                },
            }
        })
    else:
        decode_model = None
#        decode_model = "/home/eric/ray_results/iltrain/il_0_2018-03-23_21-00-01kbtfotn3/weights_51",  # oracle autoencoder
#        decode_model = "/home/eric/ray_results/iltrain/il_0_2018-03-24_15-02-186tp9e4r7/weights_20",  # ivd large dataset
#        decode_model = "/home/eric/ray_results/iltrain/il_0_2018-03-23_21-08-392dnx7np8/weights_20",  # ivd + il
#        decode_model = "/home/eric/ray_results/iltrain/il_0_2018-03-23_20-54-08gju9v3k_/weights_16",  # ivd
#        decode_model = "/home/eric/ray_results/iltrain/il_0_2018-03-23_20-36-57j4yi7nev/weights_39",  # il
        run_experiments({
            "test-cartpole": {
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
