from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.a3c.a3c import A3CAgent, DEFAULT_CONFIG as A3C_CONFIG
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.utils import merge_dicts

A2C_DEFAULT_CONFIG = merge_dicts(
    A3C_CONFIG,
    {
        "sample_batch_size": 100,
        "min_iter_time_s": 10,
        "optimizer": {
            "timesteps_per_batch": 1000,
        },
    },
)


class A2CAgent(A3CAgent):
    """Synchronous variant of the A3CAgent."""

    _agent_name = "A2C"
    _default_config = A2C_DEFAULT_CONFIG

    def _make_optimizer(self):
        return SyncSamplesOptimizer(self.local_evaluator,
                                    self.remote_evaluators,
                                    self.config["optimizer"])
