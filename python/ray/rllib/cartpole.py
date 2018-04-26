from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from collections import deque
import math
import numpy as np
import json
import pickle
import gym
from gym import spaces
import random
import scipy.stats
import time
import tensorflow as tf

import ray
from ray.rllib.carracing_discrete import env_test
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.models.visionnet import VisionNetwork
from ray.tune import run_experiments, grid_search, register_env
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent
#from ray.rllib import bullet_cartpole
from ray.rllib.render_cartpole import render_frame
from ray.rllib import a3c
from ray.rllib.carracing_discrete.atari_wrappers import NoopResetEnv, WarpFrame, FrameStack
from ray.rllib.carracing_discrete.wrapper import DiscreteCarRacing
from ray.rllib.utils.compression import unpack

parser = argparse.ArgumentParser()
parser.add_argument("--car", action="store_true")
parser.add_argument("--image", action="store_true")
parser.add_argument("--decode-model", default=None)
parser.add_argument("--background", default="noise")
parser.add_argument("--experiment", default="cartpole-decode")
parser.add_argument("--dataset", default=None)
parser.add_argument("--pca", action="store_true")
parser.add_argument("--h-size", default=32, type=int)
parser.add_argument("--num-snow", default=30, type=int)

def framestack_cartpole(data, k, env_config, args):
    frames = deque([], maxlen=k)
    data_out = []
    for t in data:
        ok = len(frames) >= k
        if len(frames) == 0:
            frames.append(render_frame(t["obs"], env_config))
        if ok:
            t["encoded_obs"] = np.concatenate(frames, axis=2)
        frames.append(render_frame(t["new_obs"], env_config))
        if ok:
            t["encoded_next_obs"] = np.concatenate(frames, axis=2)
            data_out.append(t)
        if t["done"]:
            frames.clear()
        if len(data_out) % 1000 == 0:
            print("Loaded frames", len(data_out))
    return data_out


def load_images(data, args, env_config):
    data = [json.loads(x) for x in open(data).readlines()]
    if args.car:
        render_f = lambda args: args[0]
    else:
        render_f = render_frame
    if args.car:
        for d in data:
            d["encoded_obs"] = d["obs"]
    else:
        data = framestack_cartpole(data, 4, env_config, args)
    return np.stack(x["encoded_obs"].flatten() for x in data)


def build_racing_env(_):
    env = gym.make('CarRacing-v0')
    env = DiscreteCarRacing(env)
    env = NoopResetEnv(env)
    env.override_num_noops = 50
    env = WarpFrame(env, 80)
    env = FrameStack(env, 4)
    # hack needed to fix rendering on CarRacing-v0
    env = gym.wrappers.Monitor(env, "/tmp/rollouts", resume=True)
    return env



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


class PCADecoder(Preprocessor):
    def _init(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        self.h_size = self._options["custom_options"].get("h_size")
        dataset = self._options["custom_options"].get("pca_train")
        env_config = self._options["custom_options"].get("env_config")
        self.pca = PCA(self.h_size)
        scaler = StandardScaler()
        data = load_images(dataset, args, env_config)
        s = scaler.fit(data)
        data = s.transform(data)
        print("Fitting PCA on data", data.shape)
        self.pca.fit(data)
        print("N PCA components", self.pca.n_components_)
        self.shape = (self.h_size,)

    def transform(self, obs):
        return self.pca.transform([obs.flatten()])[0][:self.h_size]


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
    ModelCatalog.register_custom_preprocessor("pca_decoder", PCADecoder)
    register_env(
        "ImageCartPole-v0",
        lambda env_config: ImageCartPole(gym.make("CartPole-v0"), 4, env_config))
#    register_env(
#        "BulletCartPole-v0",
#        lambda config: bullet_cartpole.BulletCartpole(
#            bullet_cartpole.add_opts(
#                argparse.ArgumentParser()).parse_args([]),
#            discrete_actions=True))
    
    env_creator_name = "discrete-carracing-v0"
    register_env(env_creator_name, build_racing_env)

    ray.init()
    args = parser.parse_args()
    env_config = {
        "background": args.background,
    }

    decode_model = args.decode_model and os.path.expanduser(args.decode_model)
    if args.image or args.car:
        if args.pca:
            model_opts = {
                "custom_preprocessor": "pca_decoder",
                "custom_options": {
                    "h_size": args.h_size,
                    "pca_train": args.dataset,
                    "env_config": env_config,
                },
            }
        elif decode_model:
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
                    "run": "A3C",
                    "env": "discrete-carracing-v0",
                    "repeat": 1,
                    "trial_resources": {
                        "cpu": 1,
                        "gpu": 0,
                        "extra_cpu": lambda spec: spec.config.num_workers,
                    },
                    "config": {
                        "num_workers": 7,
                        "optimizer": {
                            "grads_per_step": 1000    
                        },
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
                        "gpu": 0,
                        "extra_cpu": lambda spec: spec.config.num_workers,
                    },
                    "stop": {
                        "episode_reward_mean": 200,
                    },
                    "config": {
                        "env_config": env_config,
                        "devices": ["/gpu:0"],
                        "num_sgd_iter": 10,
                        "num_workers": 1,
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
