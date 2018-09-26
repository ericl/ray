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
        name = "atari-cbp-ex120"
        scheduler = CheckpointBasedPruning(
            reltime_attr="time_since_restore",
            reward_attr="episode_reward_mean",
            checkpoint_eval_t=300,
            checkpoint_min_reward=9999,
            bootstrap_checkpoint=
#            "/home/ubuntu/ray_results/atari-checkpoints/IMPALA_BreakoutNoFrameskip-v4_0_env=BreakoutNoFrameskip-v4_2018-09-23_02-13-09cvQB93/checkpoint-40",
#"/home/ubuntu/ray_results/atari-checkpoints/IMPALA_QbertNoFrameskip-v4_2_env=QbertNoFrameskip-v4_2018-09-23_02-13-09llKPVp/checkpoint-40",
#"/home/ubuntu/ray_results/atari-checkpoints/IMPALA_SpaceInvadersNoFrameskip-v4_3_env=SpaceInvadersNoFrameskip-v4_2018-09-23_02-13-09iA5hAc/checkpoint-40",
"/home/ubuntu/ray_results/atari-fishing/IMPALA_FishingDerby-v4_10_entropy_coeff=-0.01,env=FishingDerby-v4,grad_clip=400.0,lr=0.0001,train_batch_size=500_2018-09-25_03-45-00iSNv2c/checkpoint-120",
#"/home/ubuntu/ray_results/atari-checkpoints/IMPALA_BeamRiderNoFrameskip-v4_1_env=BeamRiderNoFrameskip-v4_2018-09-23_02-13-09KUM_OF/checkpoint-40",
            reduction_factor=100)
        algo = None

    run_experiments(
        {
            name: {
                "trial_resources": {
                    "gpu": 0.5,
                    "cpu": 1,
                    "extra_cpu": 16,
                },
                "run": "IMPALA",
                "env": "FishingDerbyNoFrameskip-v4",
                "stop": {
#                    "episode_reward_mean": 20,
                    "time_total_s": 900,
                },
                "config": {
                    "num_workers": 16,
                    "clip_rewards": True,
                    "sample_batch_size": 500,
                    "train_batch_size": grid_search([500, 2000]),
                    "num_envs_per_worker": 10,
                    "lr": grid_search([0.0005, 0.0001, 0.00005, 0.00001]),
                    "grad_clip": grid_search([40.0, 400.0]),
                    "entropy_coeff": grid_search([-0.05, -0.01, -0.001]),
                },
            },
        },
        scheduler=scheduler,
        search_alg=algo)
