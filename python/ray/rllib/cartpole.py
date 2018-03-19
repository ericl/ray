from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import gym
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
    ModelCatalog.register_custom_preprocessor("my_prep", MakeCartpoleHarder)

    ray.init()
    run_experiments({
        "cartpole-decoder": {
            "run": "PPO",
            "env": "CartPole-v0",
            "repeat": 1,
            "resources": {
                "cpu": 3,
            },
            "stop": {
                "episode_reward_mean": 200,
                "timesteps_total": 200000,
#            "time_total_s": 300,
            },
            "config": {
                "num_sgd_iter": 10,
                "num_workers": 2,
                "model": {
                    "custom_preprocessor": "my_prep",
                    "custom_options": {
                        "seed": 0,
                        "noise_size": 500,  #grid_search([0, 10, 50, 100, 500, 1000]),
                        "matrix_size": 500,
                        "invert": False,
                        "decode_model": "/home/eric/Desktop/hybrid_148",
                    },
                },
            },
        }
    })
