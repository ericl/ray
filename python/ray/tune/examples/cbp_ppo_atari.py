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
    ray.init()

    args = parser.parse_args()
    if args.random:
        name = "pendulum-cbp-grid"
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
        name = "jamesbond-cbp-ex40"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=300,
            checkpoint_min_reward=99999,
            # 40 -> -831
            # 120 -> -195
            bootstrap_checkpoint=
            "/home/ubuntu/ray_results/pendulum-ppo/PPO_Pendulum-v0_0_2018-09-21_21-47-15yGq5N9/checkpoint-40",
            reduction_factor=999999)
        algo = None

    run_experiments(
        {
            name: {
                "run": "PPO",
                "env": "JamesBondNoFrameskip-v4",
                "stop": {
                    "time_total_s": 3600,
                },
                "config": {
                    "lambda": 0.95,
                    "kl_coeff": 0.5,
                    "clip_rewards": True,
                    "clip_param": grid_search([0.1, 0.3]),
                    "vf_clip_param": grid_search([1.0, 10.0]),
                    "entropy_coeff": grid_search([0.01, 0.001]),
                    "train_batch_size": 5000,
                    "lr": grid_search([0.0005, 0.0001, 0.00005, 0.00001]),
                    "sample_batch_size": 500,
                    "sgd_minibatch_size": 500,
                    "num_sgd_iter": 10,
                    "num_workers": 10,
                    "num_envs_per_worker": 5,
                    "batch_mode": "truncate_episodes",
                    "observation_filter": "NoFilter",
                    "vf_share_layers": True,
                    "num_gpus": 1,
                },
            },
        },
        scheduler=scheduler,
        search_alg=algo)

