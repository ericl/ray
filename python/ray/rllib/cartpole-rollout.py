from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import gym
import tensorflow as tf

import ray
from ray.experimental.tfutils import TensorFlowVariables
from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.tune import run_experiments, grid_search
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.ppo import PPOAgent
from ray.rllib.cartpole import MakeCartpoleHarder

sess = tf.Session()
model_config = {
    "fcnet_activation": "relu",
    "fcnet_hiddens": [256, 8],
}
obs_ph = tf.placeholder(tf.float32, [None, 504])
network = FullyConnectedNetwork(obs_ph, 2, model_config)
vars = TensorFlowVariables(network.outputs, sess)
sess.run(tf.global_variables_initializer())
action_dist = Categorical(network.outputs)

weights_file = "/home/eric/ray_results/iltrain/il_0_2018-03-19_19-12-23ynxbcjci/weights_48"
#weights_file = "/home/eric/Desktop/il_trained_560"

with open(weights_file, "rb") as f:
    vars.set_weights(pickle.loads(f.read()))

env = gym.make("CartPole-v0")
preprocessor = MakeCartpoleHarder(env.observation_space, {
    "custom_options": {
        "seed": 0,
        "noise_size": 500,
        "matrix_size": 500,
        "invert": False,
    },
})
act = action_dist.sample()
rewards = []
for _ in range(100):
    obs = env.reset()
    reward = 0
    done = False
    while not done:
        action = sess.run(act, feed_dict={obs_ph: [preprocessor.transform(obs)]})[0]
        obs, rew, done, _ = env.step(action)
        reward += rew
    rewards.append(reward)
    print("Reward", reward, "mean", np.mean(rewards))
