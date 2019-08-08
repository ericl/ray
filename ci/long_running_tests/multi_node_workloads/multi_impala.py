# This workload tests running IMPALA with remote envs

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments

ray.init(redis_address="localhost:6379")

# Run the workload.

run_experiments({
    "impala": {
        "run": "IMPALA",
        "env": "CartPole-v0",
        "config": {
            "num_workers": 6,
            "num_gpus": 0,
            "num_envs_per_worker": 5,
            "remote_worker_envs": True,
            "remote_env_batch_wait_ms": 99999999,
            "sample_batch_size": 50,
            "train_batch_size": 100,
        },
    },
})
