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
        name = "pong-random-ex"
        scheduler = None
        algo = None
    elif args.hyperopt:
        name = "pong-hyperopt"
        space = {
            "sample_batch_size": hp.choice("sample_batch_size",
                [10, 20, 40, 80, 160, 320, 640]),
            "num_envs_per_worker": hp.choice("num_envs_per_worker",
                    [1, 2, 5, 10]),
            "lr": hp.choice("lr", 
                [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001,
                 0.000005, 0.000001, 0.0000005, 0.0000001]),
        }
        algo = HyperOptSearch(
            space,
            max_concurrent=2,
            reward_attr="episode_reward_mean")
    elif args.hyperband:
        name = "pong-hyperopt"
        algo = None
        scheduler = AsyncHyperBandScheduler(
           time_attr="time_total_s",
           reward_attr="episode_reward_mean",
           max_t=1800,
           grace_period=200,
           reduction_factor=3,
           brackets=3)
    elif args.bootstrap:
        name = "pong-cbp5"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=120,
            checkpoint_min_reward=-19.5,
            bootstrap_checkpoint=None,
            reduction_factor=10)
        algo = None
    else:
        name = "pong-cbp-ex280-anneal"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=120,
            checkpoint_min_reward=9999,
            # 0 -> use raw data
            # 80 -> -19.9
            # 140 -> -17
            # 280 -> 10
            bootstrap_checkpoint="/home/ubuntu/ray_results/pong-a3c/A3C_PongDeterministic-v4_0_2018-09-17_07-57-31OEK7hT/checkpoint-280",
            reduction_factor=100)
        algo = None

    run_experiments({
        name: {
            "run": "A3C",
            "env": "PongDeterministic-v4",
            "stop": {
                "episode_reward_mean": 20,
                "time_total_s": 1800,
            },
            "config": {
                "num_workers": 16,
                "sample_batch_size":
                    grid_search(
                        [10, 40, 160, 640]),
                "lr_schedule": [
                    [0, 0.0001],
                    [100000,
                            grid_search(
                                [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001,
                                 0.000005, 0.000001, 0.0000005, 0.0000001]),
                    ],
                ],
#                "num_envs_per_worker":
#                    grid_search([1, 2, 5, 10]),
                "observation_filter": "NoFilter",
                "preprocessor_pref": "rllib",
                "model": {
                    "use_lstm": True,
                    "conv_activation": "elu",
                    "dim": 42,
                    "grayscale": True,
                    "zero_mean": False,
                    "conv_filters": [
                        [32, [3, 3], 2],
                        [32, [3, 3], 2],
                        [32, [3, 3], 2],
                        [32, [3, 3], 2],
                    ],
                },
            },
        },
    }, scheduler=scheduler, search_alg=algo)

