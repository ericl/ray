from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray
from ray.rllib.ppo.filter import RunningStat


class Optimizer(object):
    """RLlib optimizers provide pluggable SGD strategies for RL.

    For example, AsyncOptimizer is used for A3C, and LocalMultiGpuOptimizer is
    used for PPO. These optimizers are all pluggable however, it is possible
    to mix as match as needed.

    In order for an algorithm to use an RLlib optimizer, it must implement
    the Evaluator interface and pass a number of remote Evaluators to its
    Optimizer of choice. The Optimizer uses these Evaluators to sample from the
    environment and compute model gradient updates.
    """

    def __init__(self, local_evaluator, remote_evaluators):
        self.local_evaluator = local_evaluator
        self.remote_evaluators = remote_evaluators

    def step(self):
        """Takes a logical optimization step."""

        raise NotImplementedError

    def stats(self):
        """Returns a dictionary of internal performance statistics."""

        return {}


# TODO(ekl) does this have to be provided by the evaluator
def _concat(samples):
    result = []
    for s in samples:
        result.extend(s)
    return result


class SyncLocalOptimizer(Optimizer):
    def __init__(self, local_ev, remote_ev):
        Optimizer.__init__(self, local_ev, remote_ev)
        self.sample_time = RunningStat(())
        self.grad_time = RunningStat(())
        self.update_weights_time = RunningStat(())

    def step(self):
        t0 = time.time()
        if self.remote_evaluators:
            weights = ray.put(self.local_evaluator.get_weights())
            for e in self.remote_evaluators:
                e.set_weights.remote(weights)
        self.update_weights_time.push(time.time() - t0)

        t1 = time.time()
        if self.remote_evaluators:
            samples = _concat(
                ray.get([e.sample.remote() for e in self.remote_evaluators]))
        else:
            samples = self.local_evaluator.sample()
        self.sample_time.push(time.time() - t1)

        t2 = time.time()
        grad = self.local_evaluator.compute_gradients(samples)
        self.local_evaluator.apply_gradients(grad)
        self.grad_time.push(time.time() - t2)

    def stats(self):
        return {
            "sample_time_ms": round(1000 * self.sample_time.mean, 3),
            "grad_time_ms": round(1000 * self.grad_time.mean, 3),
            "update_time_ms": round(1000 * self.update_weights_time.mean, 3),
        }
