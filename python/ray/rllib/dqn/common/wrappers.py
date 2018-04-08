from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models import ModelCatalog


def wrap_dqn(registry, env, options, random_starts):
    """Apply a common set of wrappers for DQN."""

    return ModelCatalog.get_preprocessor_as_wrapper(registry, env, options)
