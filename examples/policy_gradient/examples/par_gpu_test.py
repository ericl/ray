from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import nccl
from tensorflow.python.ops.data_flow_ops import StagingArea
import pickle
import time

from reinforce.models.visionnet import vision_net
from reinforce.distributions import Categorical
from reinforce.utils import iterate


BATCH_SIZE = 256
MAX_EXAMPLES = 1024
DEVICES = ["/cpu:0", "/cpu:1", "/cpu:2", "/cpu:3"]
NUM_DEVICES = len(DEVICES)
HAS_GPU = any(['gpu' in d for d in DEVICES])


# Pong-ram-v0
observations = tf.placeholder(tf.float32, shape=(None, 80, 80, 3))
prev_logits = tf.placeholder(tf.float32, shape=(None, 6))
actions = tf.placeholder(tf.int64, shape=(None,))

def create_loss(observations, prev_logits, actions):
  curr_logits = vision_net(observations, num_classes=6)
  curr_dist = Categorical(curr_logits)
  prev_dist = Categorical(prev_logits)
  ratio = tf.exp(curr_dist.logp(actions) - prev_dist.logp(actions))
  loss = tf.reduce_mean(ratio)
  return loss


#
# Baseline strategy - no parallelism
#

dummy_loss = create_loss(observations, prev_logits, actions)
optimizer = tf.train.AdamOptimizer(5e-5)
grad = optimizer.compute_gradients(dummy_loss)
train_op = optimizer.apply_gradients(grad)
mean_loss = tf.reduce_mean(dummy_loss)


#
# Strategy 1 - use tf.split, then average the gradients
#

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """

  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

    if grads:
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads


strategy1_losses = []
strategy1_grads = []
strategy1_split_obs = tf.split(observations, NUM_DEVICES)
strategy1_split_plog = tf.split(prev_logits, NUM_DEVICES)
strategy1_split_act = tf.split(actions, NUM_DEVICES)
with tf.variable_scope("shared_strategy1_net"):
  for i, device in enumerate(DEVICES):
    with tf.device(device):
      strategy1_losses.append(
          create_loss(
              strategy1_split_obs[i],
              strategy1_split_plog[i],
              strategy1_split_act[i]))
    tf.get_variable_scope().reuse_variables()
for loss_i, device in zip(strategy1_losses, DEVICES):
  with tf.device(device):
    strategy1_grads.append(optimizer.compute_gradients(
        loss_i, colocate_gradients_with_ops=True))
strategy1_avg_grads = average_gradients(strategy1_grads)
strategy1_train_op = optimizer.apply_gradients(strategy1_avg_grads)

#
# Strategy 2 - use tf.split, but pipeline with staging area
#

strategy2_losses = []
strategy2_grads = []
strategy2_split_obs = tf.split(observations, NUM_DEVICES)
strategy2_split_plog = tf.split(prev_logits, NUM_DEVICES)
strategy2_split_act = tf.split(actions, NUM_DEVICES)
strategy2_tuples = zip(
    strategy2_split_obs, strategy2_split_act, strategy2_split_plog)
strategy2_stage_ops = []
with tf.device("/cpu:0"):
  strategy2_stage = StagingArea(
      [observations.dtype, actions.dtype, prev_logits.dtype],
      [observations.shape, actions.shape, prev_logits.shape])
for item in strategy2_tuples:
  strategy2_stage_ops.append(strategy2_stage.put(item))
strategy2_stage_op = tf.group(*strategy2_stage_ops)
with tf.variable_scope("shared_strategy2_net"):
  for i, device in enumerate(DEVICES):
    with tf.device(device):
      obs, acts, plgs = strategy2_stage.get()
      strategy2_losses.append(create_loss(obs, plgs, acts))
    tf.get_variable_scope().reuse_variables()
for loss_i, device in zip(strategy2_losses, DEVICES):
  with tf.device(device):
    strategy2_grads.append(optimizer.compute_gradients(
        loss_i, colocate_gradients_with_ops=True))
strategy2_avg_grads = average_gradients(strategy2_grads)
strategy2_train_op = optimizer.apply_gradients(strategy2_avg_grads)

#
# Strategy 3 - like (1) but using NCCL to average the gradients
#

def sum_grad_and_var_all_reduce(grad_and_vars, devices):
  # Note that each grad_and_vars looks like the following:
  #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))

  scaled_grads = [g for _, (g, _) in zip(devices, grad_and_vars) if g is not None]
  if not scaled_grads:  # no gradient for this var
    return None
  summed_grads = nccl.all_sum(scaled_grads)

  result = []
  for d, (_, v), g in zip(devices, grad_and_vars, summed_grads):
    with tf.device(d):
      result.append((g, v))
  return result


def sum_gradients_all_reduce(tower_grads, devices):
  new_tower_grads = []
  for grad_and_vars in zip(*tower_grads):
    summed = sum_grad_and_var_all_reduce(grad_and_vars, devices)
    if summed:
      new_tower_grads.append(summed)
  return list(zip(*new_tower_grads))


if HAS_GPU:
  strategy3_losses = []
  strategy3_grads = []
  strategy3_split_obs = tf.split(observations, NUM_DEVICES)
  strategy3_split_plog = tf.split(prev_logits, NUM_DEVICES)
  strategy3_split_act = tf.split(actions, NUM_DEVICES)
  with tf.variable_scope("shared_strategy3_net"):
    for i, device in enumerate(DEVICES):
      with tf.device(device):
        strategy3_losses.append(
            create_loss(
                strategy3_split_obs[i],
                strategy3_split_plog[i],
                strategy3_split_act[i]))
      tf.get_variable_scope().reuse_variables()
  for loss_i, device in zip(strategy3_losses, DEVICES):
    with tf.device(device):
      strategy3_grads.append(optimizer.compute_gradients(
          loss_i, colocate_gradients_with_ops=True))
  strategy3_avg_grads = sum_gradients_all_reduce(strategy3_grads, DEVICES)
  # Since the model is shared, applying just the 1st is sufficient
  strategy3_train_op = optimizer.apply_gradients(strategy3_avg_grads[0])


#
# Common main
#

def truncate(trajectory, size):
  batch = dict()
  for key in trajectory:
    batch[key] = trajectory[key][:size]
  return batch


config = {
    "device_count": {"CPU": NUM_DEVICES},
    "log_device_placement": True,
}
config_proto = tf.ConfigProto(device_count={"CPU": 4})
sess = tf.Session(config=config_proto)
trajectory = truncate(
  pickle.load(open("/tmp/Pong-v0-trajectory", 'rb')), MAX_EXAMPLES)
total_examples = len(trajectory["observations"])
print("Total examples", total_examples)


def make_inputs(batch):
  return {
    observations: batch["observations"],
    actions: batch["actions"].squeeze(),
    prev_logits: batch["logprobs"]
  }


def run_experiment(strategy, name):
  sess.run(tf.global_variables_initializer())
  delta = strategy(trajectory)
  print("->", name, "examples per second", total_examples / delta)


def baseline_strategy(trajectory):
  print("Current loss", sess.run(mean_loss, feed_dict=make_inputs(trajectory)))
  start = time.time()
  for i, batch in enumerate(iterate(trajectory, BATCH_SIZE)):
    print("iterate", i)
    sess.run(train_op, feed_dict = make_inputs(batch))
  delta = time.time() - start
  print("Final loss", sess.run(mean_loss, feed_dict=make_inputs(trajectory)))
  return delta


def split_parallel_strategy(trajectory):
  print("Current loss", sess.run(
      strategy1_losses[0], feed_dict=make_inputs(trajectory)))
  start = time.time()
  for i, batch in enumerate(iterate(trajectory, BATCH_SIZE)):
    print("iterate", i)
    sess.run(strategy1_train_op, feed_dict = make_inputs(batch))
  delta = time.time() - start
  print("Final loss", sess.run(
      strategy1_losses[0], feed_dict=make_inputs(trajectory)))
  return delta


def split_parallel_pipelined_strategy(trajectory):
  sess.run(strategy2_stage_op, feed_dict = make_inputs(trajectory))
  print("Current loss", sess.run(strategy2_losses[0]))
  start = time.time()
  for i, batch in enumerate(iterate(trajectory, BATCH_SIZE)):
    print("iterate", i)
    if i == 0:
      sess.run(strategy2_stage_op, feed_dict = make_inputs(batch))
    else:
      sess.run(
          [strategy2_train_op, strategy2_stage_op],
          feed_dict = make_inputs(batch))
  delta = time.time() - start
  sess.run(strategy2_train_op)
  sess.run(strategy2_stage_op, feed_dict = make_inputs(trajectory))
  print("Final loss", sess.run(strategy2_losses[0]))
  return delta


def split_parallel_nccl_strategy(trajectory):
  print("Current loss", sess.run(
      strategy3_losses[0], feed_dict=make_inputs(trajectory)))
  start = time.time()
  for i, batch in enumerate(iterate(trajectory, BATCH_SIZE)):
    print("iterate", i)
    sess.run(strategy3_train_op, feed_dict = make_inputs(batch))
  delta = time.time() - start
  print("Final loss", sess.run(
      strategy3_losses[0], feed_dict=make_inputs(trajectory)))
  return delta


run_experiment(baseline_strategy, "Baseline")
run_experiment(split_parallel_strategy, "Split parallel")
run_experiment(split_parallel_pipelined_strategy, "Split parallel pipelined")
if HAS_GPU:
  run_experiment(split_parallel_nccl_strategy, "Split parallel + NCCL")


"""
All possible decisions:

Model strategies:
    - N model graphs with separate weights
    - N model graphs with shared weights

Input strategies:
    - set_var -> tf.slice_input_producer -> tf.batch -> par { opt.compute_grads } -> average -> opt.apply_grads
    - feed_dict -> tf.split -> par { opt.compute_grads } -> average -> opt.apply_grads
    - above but w/staging area for pipelining the split

Averaging strategy:
    - nccl all_sum
    - average on cpu
"""

# TODO(ekl) collect timeline files for each experiment
# TODO(ekl) test having separate models assigned to each GPU
# TODO(ekl) test slice_input_producer + tf.batch to avoid feed_dict
#   https://github.com/tensorflow/tensorflow/issues/7817
