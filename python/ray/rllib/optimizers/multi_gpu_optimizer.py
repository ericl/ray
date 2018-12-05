from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
from collections import defaultdict
import tensorflow as tf

import ray
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.optimizers.multi_gpu_impl import LocalSyncParallelOptimizer
from ray.rllib.utils.timer import TimerStat
from ray.rllib.evaluation.sample_batch import SampleBatch, DEFAULT_POLICY_ID

logger = logging.getLogger(__name__)


class LocalMultiGPUOptimizer(PolicyOptimizer):
    """A synchronous optimizer that uses multiple local GPUs.

    Samples are pulled synchronously from multiple remote evaluators,
    concatenated, and then split across the memory of multiple local GPUs.
    A number of SGD passes are then taken over the in-memory data. For more
    details, see `multi_gpu_impl.LocalSyncParallelOptimizer`.

    This optimizer is Tensorflow-specific and require the underlying
    PolicyGraph to be a TFPolicyGraph instance that support `.copy()`.

    Note that all replicas of the TFPolicyGraph will merge their
    extra_compute_grad and apply_grad feed_dicts and fetches. This
    may result in unexpected behavior.
    """

    def _init(self,
              sgd_batch_size=128,
              num_sgd_iter=10,
              train_batch_size=1024,
              num_gpus=0,
              standardize_fields=[]):
        self.batch_size = sgd_batch_size
        self.num_sgd_iter = num_sgd_iter
        self.train_batch_size = train_batch_size
        if not num_gpus:
            self.devices = ["/cpu:0"]
        else:
            self.devices = [
                "/gpu:{}".format(i) for i in range(int(math.ceil(num_gpus)))
            ]
        self.batch_size = int(sgd_batch_size / len(self.devices)) * len(
            self.devices)
        assert self.batch_size % len(self.devices) == 0
        assert self.batch_size >= len(self.devices), "batch size too small"
        self.per_device_batch_size = int(self.batch_size / len(self.devices))
        self.sample_timer = TimerStat()
        self.load_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.update_weights_timer = TimerStat()
        self.standardize_fields = standardize_fields

        logger.info("LocalMultiGPUOptimizer devices {}".format(self.devices))

        self.policies = self.local_evaluator.policy_map
        for policy_id, policy in self.policies.items():
            if not isinstance(policy, TFPolicyGraph):
                raise ValueError(
                    "Only TF policies are supported with multi-GPU. Try using "
                    "the simple optimizer instead.")

        # per-GPU graph copies created below must share vars with the policy
        # reuse is set to AUTO_REUSE because Adam nodes are created after
        # all of the device copies are created.
        self.optimizers = {}
        with self.local_evaluator.tf_sess.graph.as_default():
            with self.local_evaluator.tf_sess.as_default():
                for policy_id, policy in self.policies.items():
                    with tf.variable_scope(policy_id, reuse=tf.AUTO_REUSE):
                        if policy._state_inputs:
                            rnn_inputs = policy._state_inputs + [
                                policy._seq_lens
                            ]
                        else:
                            rnn_inputs = []
                        self.optimizers[policy_id] = (
                            LocalSyncParallelOptimizer(
                                policy.optimizer(), self.devices,
                                [v
                                 for _, v in policy.loss_inputs()], rnn_inputs,
                                self.per_device_batch_size, policy.copy))

                self.sess = self.local_evaluator.tf_sess
                self.sess.run(tf.global_variables_initializer())

    def step(self):
        with self.update_weights_timer:
            if self.remote_evaluators:
                weights = ray.put(self.local_evaluator.get_weights())
                for e in self.remote_evaluators:
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.remote_evaluators:
                # TODO(rliaw): remove when refactoring
                from ray.rllib.agents.ppo.rollout import collect_samples
                samples = collect_samples(self.remote_evaluators,
                                          self.train_batch_size)
            else:
                samples = self.local_evaluator.sample()
            # Handle everything as if multiagent
            if isinstance(samples, SampleBatch):
                samples = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)

        for _, batch in samples.policy_batches.items():
            for field in self.standardize_fields:
                value = batch[field]
                standardized = (value - value.mean()) / max(1e-4, value.std())
                batch[field] = standardized

        for policy_id, policy in self.policies.items():
            # Important: don't shuffle RNN sequence elements
            if (policy_id in samples.policy_batches
                    and not policy._state_inputs):
                samples.policy_batches[policy_id].shuffle()

        num_loaded_tuples = {}
        with self.load_timer:
            for policy_id, batch in samples.policy_batches.items():
                policy = self.policies[policy_id]
                tuples = policy._get_loss_inputs_dict(batch)
                data_keys = [ph for _, ph in policy.loss_inputs()]
                if policy._state_inputs:
                    state_keys = policy._state_inputs + [policy._seq_lens]
                else:
                    state_keys = []
                num_loaded_tuples[policy_id] = (
                    self.optimizers[policy_id].load_data(
                        self.sess, [tuples[k] for k in data_keys],
                        [tuples[k] for k in state_keys]))

        fetches = {}
        with self.grad_timer:
            for policy_id, tuples_per_device in num_loaded_tuples.items():
                optimizer = self.optimizers[policy_id]
                num_batches = (
                    int(tuples_per_device) // int(self.per_device_batch_size))
                logger.debug("== sgd epochs for {} ==".format(policy_id))
                for i in range(self.num_sgd_iter):
                    iter_extra_fetches = defaultdict(list)
                    permutation = np.random.permutation(num_batches)
                    for batch_index in range(num_batches):
                        batch_fetches = optimizer.optimize(
                            self.sess, permutation[batch_index] *
                            self.per_device_batch_size)
                        for k, v in batch_fetches.items():
                            iter_extra_fetches[k].append(v)
                    logger.debug("{} {}".format(i,
                                                _averaged(iter_extra_fetches)))
                fetches[policy_id] = _averaged(iter_extra_fetches)

        self.num_steps_sampled += samples.count
        self.num_steps_trained += samples.count
        return fetches

    def stats(self):
        return dict(
            PolicyOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "load_time_ms": round(1000 * self.load_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
            })


def _averaged(kv):
    out = {}
    for k, v in kv.items():
        if v[0] is not None:
            out[k] = np.mean(v)
    return out
