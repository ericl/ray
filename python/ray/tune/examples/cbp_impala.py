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
    ray.init(redis_address="localhost:6379")

    args = parser.parse_args()
    if args.random:
        name = "atari-random-ex"
        scheduler = None
        algo = None
    elif args.hyperopt:
        name = "atari-hyperopt"
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
        name = "atari-hyperopt"
        algo = None
        scheduler = AsyncHyperBandScheduler(
            time_attr="time_total_s",
            reward_attr="episode_reward_mean",
            max_t=1800,
            grace_period=200,
            reduction_factor=3,
            brackets=3)
    elif args.bootstrap:
        name = "atari-cbp5"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=120,
            checkpoint_min_reward=-19.5,
            bootstrap_checkpoint=None,
            reduction_factor=10)
        algo = None
    else:
        name = "atari-cbp-ex40"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=120,
            checkpoint_min_reward=9999,
            bootstrap_checkpoint=
            "/home/ubuntu/ray_results/atari-impala/IMPALA_BreakoutNoFrameskip-v4_0_env=BreakoutNoFrameskip-v4_2018-09-05_05-16-59YOnUrv/checkpoint-40",
            reduction_factor=100)
        algo = None

    run_experiments(
        {
            name: {
                "run": "IMPALA",
                "env": "BreakoutNoFrameskip-v4",
                "stop": {
#                    "episode_reward_mean": 20,
                    "time_total_s": 900,
                },
                "config": {
                    "num_workers": 32,
                    "clip_reward": True,
                    "sample_batch_size": 250,
                    "train_batch_size": 500,
                    "num_envs_per_worker": 5,
                    "lr": grid_search([0.005, 0.0005, 0.00005]),
                    "grad_clip": grid_search([2.0, 40.0, 800.0]),
                    "entropy_coeff": grid_search([-0.1, -0.01, -0.0001]),
                },
            },
        },
        scheduler=scheduler,
        search_alg=algo)
