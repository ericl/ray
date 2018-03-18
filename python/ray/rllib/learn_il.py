from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import tensorflow as tf

from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.cartpole import MakeCartpoleHarder

N = 500

observations = tf.placeholder(tf.float32, [None, N + 4])
expert_actions = tf.placeholder(tf.int32, [None])
network = FullyConnectedNetwork(observations, 2, {})
action_dist = Categorical(network.outputs)

loss = -tf.reduce_mean(action_dist.logp(expert_actions))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

env = gym.make("CartPole-v0")
preprocessor = MakeCartpoleHarder(env.observation_space, {
    "custom_options": {
        "seed": 0,
        "noise_size": N,
        "matrix_size": N,
        "invert": False,
    },
})

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
    cur_loss, _ = sess.run([loss, train_op], feed_dict={
        observations: [
            preprocessor.transform([1, 2, 2, 1]),
            preprocessor.transform([1, 2, 3, 4])],
        expert_actions: [0, 1],
    })
    print("iteration {} accuracy {}".format(i, np.exp(-cur_loss)))
