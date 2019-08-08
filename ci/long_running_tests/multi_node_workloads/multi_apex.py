# This workload tests running APEX

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.tune import run_experiments

ray.init(redis_address="localhost:6379")

# Run the workload.

run_experiments({
    "apex": {
        "run": "APEX",
        "env": "Pong-v0",
        "config": {
            "num_workers": 6,
            "num_gpus": 0,
            "buffer_size": 10000,
            "learning_starts": 0,
            "sample_batch_size": 1,
            "train_batch_size": 1,
            "min_iter_time_s": 10,
            "timesteps_per_iteration": 10,
        },
    }
})
