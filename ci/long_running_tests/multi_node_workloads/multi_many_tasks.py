# This workload tests submitting and getting many tasks over and over.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray

ray.init(redis_address="localhost:6379")

# Run the workload.


@ray.remote
def f(*xs):
    return 1


iteration = 0
ids = []
start_time = time.time()
previous_time = start_time
while True:
    for _ in range(50):
        new_constrained_ids = [
            f._remote(args=[*ids], resources={str(i % num_nodes): 1})
            for i in range(25)
        ]
        new_unconstrained_ids = [f.remote(*ids) for _ in range(25)]
        ids = new_constrained_ids + new_unconstrained_ids

    ray.get(ids)

    new_time = time.time()
    print("Iteration {}:\n"
          "  - Iteration time: {}.\n"
          "  - Absolute time: {}.\n"
          "  - Total elapsed time: {}.".format(
              iteration, new_time - previous_time, new_time,
              new_time - start_time))
    previous_time = new_time
    iteration += 1
