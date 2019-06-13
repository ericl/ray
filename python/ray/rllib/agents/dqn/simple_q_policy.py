from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Discrete
import numpy as np
from scipy.stats import entropy

import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models import ModelCatalog, Categorical
from ray.rllib.models.modelv2 import OutputSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.tf_policy import TFPolicy, \
    LearningRateSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"


class QLoss(object):
    def __init__(self,
                 q_t_selected,
                 q_tp1_best,
                 rewards,
                 done_mask,
                 gamma=0.99):
        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rewards + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        self.td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        self.loss = tf.reduce_mean(_huber_loss(self.td_error))


class QNetwork(object):
    def __init__(self,
                 model,
                 obs,
                 is_training,
                 num_actions,
                 hiddens=[256]):
        self.model = model
        model_outputs, feature_layer, state = model({
            "obs": obs,
            "is_training": is_training,
        }, [], None)

        def build_action_value():
            # Avoid postprocessing the outputs. This enables custom models
            # to be used for parametric action DQN.
            action_out = model_outputs
            if hiddens:
                action_scores = tf.layers.dense(
                    action_out, units=num_actions, activation=None)
            else:
                action_scores = model_outputs
            return action_scores

        action_scores = model.get_branch_output(
             "action_value",
             feature_layer=feature_layer,
             default_impl=build_action_value)

        self.value = action_scores


class QValuePolicy(object):
    def __init__(self, q_values, observations, num_actions, stochastic, eps):
        deterministic_actions = tf.argmax(q_values, axis=1)
        batch_size = tf.shape(observations)[0]

        # Special case masked out actions (q_value ~= -inf) so that we don't
        # even consider them for exploration.
        random_valid_action_logits = tf.where(
            tf.equal(q_values, tf.float32.min),
            tf.ones_like(q_values) * tf.float32.min, tf.ones_like(q_values))
        random_actions = tf.squeeze(
            tf.multinomial(random_valid_action_logits, 1), axis=1)

        chose_random = tf.random_uniform(
            tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions,
                                      deterministic_actions)
        self.action = tf.cond(stochastic, lambda: stochastic_actions,
                              lambda: deterministic_actions)
        self.action_prob = None


class ExplorationStateMixin(object):
    def __init__(self, obs_space, action_space, config):
        self.cur_epsilon = 1.0
        self.stochastic = tf.placeholder(tf.bool, (), name="stochastic")
        self.eps = tf.placeholder(tf.float32, (), name="eps")

    def add_parameter_noise(self):
        if self.config["parameter_noise"]:
            self.sess.run(self.add_noise_op)

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    @override(Policy)
    def get_state(self):
        return [TFPolicy.get_state(self), self.cur_epsilon]

    @override(Policy)
    def set_state(self, state):
        TFPolicy.set_state(self, state[0])
        self.set_epsilon(state[1])


class TargetNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        # update_target_fn will be called periodically to copy Q network to
        # target Q network
        update_target_expr = []
        assert len(self.q_func_vars) == len(self.target_q_func_vars), \
            (self.q_func_vars, self.target_q_func_vars)
        for var, var_target in zip(self.q_func_vars, self.target_q_func_vars):
            update_target_expr.append(var_target.assign(var))
        self.update_target_expr = tf.group(*update_target_expr)

    def update_target(self):
        return self.get_session().run(self.update_target_expr)


def build_q_model(policy, obs_space, action_space, config):

    if not isinstance(action_space, Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    policy.q_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        OutputSpec(action_space.n),
        config["model"],
        framework="tf",
        name=Q_SCOPE)

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        OutputSpec(action_space.n),
        config["model"],
        framework="tf",
        name=Q_TARGET_SCOPE)

    return policy.q_model


def build_q_networks(policy, q_model, input_dict, obs_space, action_space,
                     config):

    # Action Q network
    q_values, _ = _q_network_from(
        policy, q_model, input_dict[SampleBatch.CUR_OBS], obs_space,
        action_space)
    policy.q_values = q_values
    policy.q_func_vars = q_model.variables()

    # Action outputs
    qvp = QValuePolicy(q_values, input_dict[SampleBatch.CUR_OBS],
                       action_space.n, policy.stochastic, policy.eps)
    policy.output_actions, policy.action_prob = qvp.action, qvp.action_prob

    return policy.output_actions, policy.action_prob


def build_q_losses(policy, batch_tensors):
    # q network evaluation
    q_t, model = _q_network_from(
        policy, policy.q_model, batch_tensors[SampleBatch.CUR_OBS],
        policy.observation_space, policy.action_space)

    # target q network evalution
    q_tp1, _ = _q_network_from(
        policy, policy.target_q_model, batch_tensors[SampleBatch.NEXT_OBS],
        policy.observation_space, policy.action_space)
    policy.target_q_func_vars = policy.target_q_model.variables()

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = tf.one_hot(
        tf.cast(batch_tensors[SampleBatch.ACTIONS], tf.int32),
        policy.action_space.n)
    q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)

    # compute estimate of best possible value starting from state at t + 1
    q_tp1_best_one_hot_selection = tf.one_hot(
        tf.argmax(q_tp1, 1), policy.action_space.n)
    q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)

    policy.q_loss = _build_q_loss(
        q_t_selected, q_tp1_best,
        batch_tensors[SampleBatch.REWARDS], batch_tensors[SampleBatch.DONES],
        policy.config)

    return policy.q_loss.loss


def _q_network_from(policy, model, obs, obs_space, action_space):
    config = policy.config
    qnet = QNetwork(model, obs, policy._get_is_training_placeholder(),
                    action_space.n, config["hiddens"])
    return qnet.value, qnet.model


def _build_q_loss(q_t_selected, q_tp1_best, rewards, dones, config):
    return QLoss(q_t_selected, q_tp1_best, rewards,
                 tf.cast(dones, tf.float32), config["gamma"])


def _huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))


def exploration_setting_inputs(policy):
    return {
        policy.stochastic: True,
        policy.eps: policy.cur_epsilon,
    }


def setup_early_mixins(policy, obs_space, action_space, config):
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ExplorationStateMixin.__init__(policy, obs_space, action_space, config)


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


SimpleQPolicy = build_tf_policy(
    name="SimpleQPolicy",
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=build_q_model,
    action_sampler_fn=build_q_networks,
    loss_fn=build_q_losses,
    extra_action_feed_fn=exploration_setting_inputs,
    extra_action_fetches_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    before_init=setup_early_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        ExplorationStateMixin,
        TargetNetworkMixin,
        LearningRateSchedule,
    ])
