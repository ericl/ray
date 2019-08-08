# This workload tests running PBT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments
from ray.tune.schedulers import PopulationBasedTraining

ray.init(redis_address="localhost:6379")

# Run the workload.

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="episode_reward_mean",
    mode="max",
    perturbation_interval=10,
    hyperparam_mutations={
        "lr": [0.1, 0.01, 0.001, 0.0001],
    })

run_experiments(
    {
        "pbt_test": {
            "run": "PG",
            "env": "CartPole-v0",
            "num_samples": 6,
            "config": {
                "lr": 0.01,
            },
        }
    },
    scheduler=pbt,
    verbose=False)
