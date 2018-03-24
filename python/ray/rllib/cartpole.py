from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import pickle
import gym
import random
import scipy.stats
import time
import tensorflow as tf

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.tune import run_experiments, grid_search
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def logit(x):
  return np.log(x) - np.log(1 - x)


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
#            assert mean >= 0 and mean <= self.elem_size, (mean, i, obs[i])
            elem = [
                normpdf(j, mean, std) + math.sin(j + mean) #+ random.random()
                for j in range(self.elem_size)
            ]
            out.extend(elem)
        if self.decode_model:
            out = self.sess.run(self.decoder.last_layer, feed_dict={
                self.obs_ph: [out]
            })[0]
#        print(obs)
#        print(means)
#        print()
#        print(out)
        return out


class MakeCartpoleHarder(Preprocessor):
    def _init(self):
        self.shape = self._obs_space.shape
        np.random.seed(self._options["custom_options"]["seed"])
        self.noise_size = self._options["custom_options"]["noise_size"]
        self.matrix_size = self._options["custom_options"]["matrix_size"]
        self.invert = self._options["custom_options"].get("invert", False)
        self.decode_model = self._options["custom_options"].get("decode_model")
        self.A = np.random.rand(
            4 + self.matrix_size, 4 + self.noise_size) - 0.5
        if self.invert:
            self.shape = (4,)
            self.A_inv = np.linalg.inv(self.A)
        else:
            if self.decode_model:
                self.sess = tf.Session()
                self.obs_ph = tf.placeholder(
                    tf.float32, [None, 4 + self.matrix_size])
                self.decoder = load_model(self.decode_model, self.obs_ph, self.sess)
                self.shape = (8,)
            else:
                self.shape = (4 + self.matrix_size,)

    def transform(self, observation):
        noise = np.random.rand(self.noise_size)
        y = np.concatenate([observation, noise])
        tmp = np.matmul(self.A, y)
        out = sigmoid(tmp)
        if self.invert:
            return self.invert_transform(out)
        elif self.decode_model:
            dec = self.sess.run(self.decoder.last_layer, feed_dict={
                self.obs_ph: [out]
            })[0]
            print(dec)
            return dec
        else:
            return out

    def invert_transform(self, observation):
        """Invert the transform for testing"""

        tmp = logit(observation)
        y = np.matmul(self.A_inv, tmp)
        orig = y[:4]
        return orig


if __name__ == '__main__':
#    ModelCatalog.register_custom_preprocessor("my_prep", MakeCartpoleHarder)
    ModelCatalog.register_custom_preprocessor("encoder", CartpoleEncoder)

    ray.init()
    run_experiments({
        "ivd-large-decoder-model": {
            "run": "PPO",
            "env": "CartPole-v0",
            "repeat": 1,
            "resources": {
                "cpu": 3,
            },
            "stop": {
                "episode_reward_mean": 200,
                "timesteps_total": 500000,
#            "time_total_s": 300,
            },
            "config": {
                "num_sgd_iter": 10,
                "num_workers": 1,
                "model": {
                    "custom_preprocessor": "encoder",
                    "custom_options": {
                        "seed": 0,
                        "out_size": grid_search([200]),
#                        "noise_size": 500,  #grid_search([0, 10, 50, 100, 500, 1000]),
#                        "matrix_size": 500,
#                        "invert": False,
#                         "decode_model": "/home/eric/ray_results/iltrain/il_0_2018-03-23_21-00-01kbtfotn3/weights_51",  # oracle autoencoder
#                         "decode_model": "/home/eric/ray_results/iltrain/il_0_2018-03-24_15-02-186tp9e4r7/weights_20",  # ivd large dataset
#                         "decode_model": "/home/eric/ray_results/iltrain/il_0_2018-03-23_21-08-392dnx7np8/weights_20",  # ivd + il
                        "decode_model": "/home/eric/ray_results/iltrain/il_0_2018-03-23_20-54-08gju9v3k_/weights_16",  # ivd
#                        "decode_model": "/home/eric/ray_results/iltrain/il_0_2018-03-23_20-36-57j4yi7nev/weights_39",  # il
                    },
                },
            },
        }
    })
