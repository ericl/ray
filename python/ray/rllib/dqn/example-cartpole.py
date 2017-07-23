#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.dqn import DQN, DEFAULT_CONFIG


def main():
    config = DEFAULT_CONFIG.copy()
    config.update(dict(
        lr=1e-3,
        schedule_max_timesteps=100000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        dueling=False,
        double_q=False,
        hiddens=[],
        prioritized_replay=False,
        model_config=dict(
            fcnet_hiddens=[64],
            fcnet_activation=tf.nn.relu
        )))

    dqn = DQN("CartPole-v0", config)

    while True:
        res = dqn.train()
        print("current status: {}".format(res))


if __name__ == '__main__':
    main()
