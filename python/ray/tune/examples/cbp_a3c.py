import numpy as np
import os

import ray
from ray.tune import Trainable, run_experiments, grid_search
from ray.tune.suggest import HyperOptSearch
from ray.tune.schedulers import CheckpointBasedPruning, AsyncHyperBandScheduler
#from hyperopt import hp


if __name__ == "__main__":
    ray.init(num_cpus=40)

#    space = {
#        "a_0": hp.uniform("a_0", 0, 1),
#        "a_1": hp.uniform("a_1", 0, 1),
#        "a_2": hp.uniform("a_2", 0, 1),
#    }
#    algo = HyperOptSearch(
#        space,
#        max_concurrent=4,
#        reward_attr="episode_reward_mean")
    
    cbp = CheckpointBasedPruning(
        reltime_attr="time_since_restore",
        reward_attr="episode_reward_mean",
        checkpoint_eval_t=120,
        checkpoint_min_reward=9999,
        bootstrap_checkpoint="/home/ubuntu/ray_results/pong-a3c/A3C_PongDeterministic-v4_0_2018-09-17_07-57-31OEK7hT/checkpoint-140",
        reduction_factor=10)
    
#    hb = AsyncHyperBandScheduler(
#       time_attr="training_iteration",
#       reward_attr="episode_reward_mean",
#       max_t=100,
#       grace_period=10,
#       reduction_factor=3,
#       brackets=3)

    run_experiments({
        "pong-cpb": {
            "run": "A3C",
            "env": "PongDeterministic-v4",
            "stop": {
                "episode_reward_mean": 20,
                "time_total_s": 900,
            },
            "config": {
                "num_workers": 16,
                "sample_batch_size":
                    grid_search(
                        [10, 20, 40, 80, 160, 320, 640]),
                "lr":
                    grid_search(
                        [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001,
                         0.000005, 0.000001, 0.0000005, 0.0000001]),
                "num_envs_per_worker":
                    grid_search([1, 2, 5, 10]),
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
    }, scheduler=cbp) #search_alg=algo) #scheduler=cbp)

