import numpy as np
import os
import argparse

import ray
from ray.tune import Trainable, run_experiments, grid_search
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import CheckpointBasedPruning, AsyncHyperBandScheduler
#from hyperopt import hp

parser = argparse.ArgumentParser()
parser.add_argument("--random", action="store_true")
parser.add_argument("--hyperopt", action="store_true")
parser.add_argument("--hyperband", action="store_true")
parser.add_argument("--bootstrap", action="store_true")

if __name__ == "__main__":
    ray.init(num_cpus=40)

    args = parser.parse_args()
    if args.random:
        name = "pendulum-cbp-random"
        scheduler = None
        algo = None
    elif args.hyperopt:
        name = "pendulum-hyperopt"
        space = {
            "sample_batch_size": hp.choice("sample_batch_size",
                                           [10, 20, 40, 80, 160, 320, 640]),
            "num_envs_per_worker": hp.choice("num_envs_per_worker",
                                             [1, 2, 5, 10]),
            "lr": hp.choice("lr", [
                0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005,
                0.000001, 0.0000005, 0.0000001
            ]),
        }
        algo = HyperOptSearch(
            space, max_concurrent=2, reward_attr="episode_reward_mean")
    elif args.hyperband:
        name = "pendulum-hyperopt"
        algo = None
        scheduler = AsyncHyperBandScheduler(
            time_attr="time_total_s",
            reward_attr="episode_reward_mean",
            max_t=300,
            grace_period=30,
            reduction_factor=3,
            brackets=3)
    elif args.bootstrap:
        name = "pendulum-cbp5"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=30,
            checkpoint_min_reward=-900,
            bootstrap_checkpoint=None,
            reduction_factor=10)
        algo = None
    else:
        name = "pendulum-cbp5"
        assert False, "No bootstrap checkpoint"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=30,
            checkpoint_min_reward=-900,
            bootstrap_checkpoint=None,
            reduction_factor=10)
        algo = None

    run_experiments(
        {
            name: {
                "run": "PPO",
                "env": "Pendulum-v0",
                "stop": {
                    "time_total_s": 300,
                },
                "config": {
                    "num_workers": 4,
                    "gamma": 0.95,
                    "lambda": 0.1,
                    "sample_batch_size": 100,
                    "num_sgd_iter": grid_search([1, 2, 4, 8, 16]),
                    "train_batch_size": grid_search(
                        [400, 800, 1600, 3200, 6400]),
                    "sgd_minibatch_size": grid_search([50, 100, 200, 400]),
                    "lr": grid_search(
                        [0.001, 0.0005, 0.0003, 0.0001, 0.00005, 0.00001]),
                    "num_envs_per_worker": grid_search([1, 2, 5, 10]),
                },
            },
        },
        scheduler=scheduler,
        search_alg=algo)
