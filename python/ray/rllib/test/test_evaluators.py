from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import gym
import shutil
import tempfile
import time

import ray
from ray.rllib.a3c import DEFAULT_CONFIG
from ray.rllib.a3c.a3c_tf_policy import A3CTFPolicyLoss
from ray.rllib.dqn.dqn_policy_loss import adjust_nstep
from ray.tune.registry import get_registry


class DQNEvaluatorTest(unittest.TestCase):
    def testNStep(self):
        obs = [1, 2, 3, 4, 5, 6, 7]
        actions = ["a", "b", "a", "a", "a", "b", "a"]
        rewards = [10.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100000.0]
        new_obs = [2, 3, 4, 5, 6, 7, 8]
        dones = [1, 0, 0, 0, 0, 1, 0]
        adjust_nstep(3, 0.9, obs, actions, rewards, new_obs, dones)
        self.assertEqual(obs, [1, 2, 3, 4, 5])
        self.assertEqual(actions, ["a", "b", "a", "a", "a"])
        self.assertEqual(rewards, [10.0, 171.0, 271.0, 271.0, 190.0])
        self.assertEqual(new_obs, [2, 5, 6, 7, 7])
        self.assertEqual(dones, [1, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
