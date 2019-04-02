#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import json
import os
import random
import time

import ray
from ray.tune import Trainable, run, sample_from
from ray.tune.schedulers import PopulationBasedTraining


class MyTrainableClass(Trainable):
    """Fake agent whose learning rate is determined by dummy factors."""

    def _setup(self, config):
        self.lr = config["lr"]
        self.score = 0.0  # end = 1000

    def _train(self):
        midpoint = 100
        q_tolerance = 3
        noise_level = 2
        # triangle wave:
        #  - start at 0.001 @ t=0,
        #  - peak at 0.01 @ t=midpoint,
        #  - end at 0.001 @ t=midpoint * 2,
        if self.score < midpoint:
            optimal_lr = 0.01 * self.score / midpoint
        else:
            optimal_lr = 0.01 - 0.01 * (self.score - midpoint) / midpoint
        optimal_lr = min(0.01, max(0.001, optimal_lr))

        q_err = max(self.lr, optimal_lr) / min(self.lr, optimal_lr)
        if q_err < q_tolerance:
            self.score += (1.0 / q_err) * random.random()
        elif self.lr > optimal_lr:
            self.score -= (q_err - q_tolerance) * random.random()
        self.score += noise_level * np.random.normal()
        self.score = max(0, self.score)

        return {
            "episode_reward_mean": self.score,
            "cur_lr": self.lr,
            "optimal_lr": optimal_lr,
            "q_err": q_err,
            "done": self.score > midpoint * 2,
        }

    def _save(self, checkpoint_dir):
        return {
            "score": self.score,
            "lr": self.lr,
        }

    def _restore(self, checkpoint):
        self.score = checkpoint["score"]

    def reset_config(self, new_config):
        self.lr = new_config["lr"]
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=4)  # force pausing to happen for test
    else:
        ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=20,
        hyperparam_mutations={
            "lr": lambda: random.uniform(0.0001, 0.02),
        })

    # Try to find the best factor 1 and factor 2
    run(MyTrainableClass,
        name="pbt_test2",
        scheduler=pbt,
        reuse_actors=True,
        verbose=False,
        **{
            "stop": {
                "training_iteration": 2000,
            },
            "num_samples": 2,
            "config": {
                "lr": 0.0001,
            },
        })

#    run(MyTrainableClass,
#        name="grid",
#        verbose=False,
#        **{
#            "stop": {
#                "training_iteration": 2000,
#            },
#            "num_samples": 1,
#            "config": {
#                "lr": 0.001,
#            },
#        })
