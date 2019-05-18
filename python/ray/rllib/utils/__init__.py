import logging
import os

from ray.rllib.utils.filter_manager import FilterManager
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.policy_client import PolicyClient
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.util import merge_dicts, deep_update

logger = logging.getLogger(__name__)


def renamed_class(cls, old_name=None):
    """Helper class for renaming classes with a warning."""

    class DeprecationWrapper(cls):
        def __init__(self, *args, **kw):
            if not old_name:
                # special case shorthand for the agent rename
                prev = cls.__name__.replace("Trainer", "Agent")
            else:
                prev = old_name
            new_name = cls.__module__ + "." + cls.__name__
            logger.warn("DeprecationWarning: {} has been renamed to {}. ".
                        format(prev, new_name) +
                        "This will raise an error in the future.")
            cls.__init__(self, *args, **kw)

    DeprecationWrapper.__name__ = cls.__name__

    return DeprecationWrapper


def try_import_tf():
    if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
        logger.warning("Not importing TensorFlow for test purposes")
        return None

    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        return tf
    except ImportError:
        try:
            import tensorflow as tf
            return tf
        except ImportError:
            return None


__all__ = [
    "Filter",
    "FilterManager",
    "PolicyClient",
    "PolicyServer",
    "merge_dicts",
    "deep_update",
    "renamed_class",
    "try_import_tf",
]
